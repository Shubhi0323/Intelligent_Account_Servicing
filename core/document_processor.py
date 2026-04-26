"""
Document Processor
──────────────────
OCR priority order (controlled by .env flags):

  1. USE_MOCK_OCR=true  → always return deterministic mock text (offline dev)
  2. USE_OCR_API=true   → call OCR.space REST API (primary production path)
  3. fallback           → return mock text if API fails / key missing

No local Tesseract installation required.

Output contract (unchanged for all downstream modules):
    {
        "text":        str   – raw extracted text
        "ocr_quality": float – 0-1 image readability estimate
        "used_mock":   bool  – True if real OCR was NOT used
        "source":      str   – "mock" | "ocr_space" | "fallback"
        "error":       str | None
    }
"""

import io
import base64
import logging
import numpy as np

import requests
from PIL import Image

from core.config import USE_MOCK_OCR, OCR_SPACE_API_KEY, USE_OCR_API

logger = logging.getLogger(__name__)

# ── OCR.space endpoint ────────────────────────────────────────────────────────
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# ── Mock text templates ───────────────────────────────────────────────────────
_MOCK_OCR_TEMPLATES: dict[str, str] = {
    "Legal Name Change": (
        "MARRIAGE CERTIFICATE\n"
        "This is to certify that the marriage between\n"
        "RAVI KUMAR SHARMA and PRIYA VERMA\n"
        "was solemnized on the 15th day of March 2022\n"
        "under the Hindu Marriage Act, 1955.\n"
        "The wife henceforth adopts the name PRIYA RAVI SHARMA.\n"
        "Registrar of Marriages, New Delhi\n"
        "Certificate No: DEL/2022/MRG/00451"
    ),
    "Address Change": (
        "ELECTRICITY BILL\n"
        "Consumer Name : RAVI KUMAR SHARMA\n"
        "Consumer No   : 1234567890\n"
        "Service Address: 42, Sector 18, Noida, Uttar Pradesh - 201301\n"
        "Bill Date      : 01-Apr-2024\n"
        "Amount Due     : INR 1,245.00\n"
        "UPPCL | Uttar Pradesh Power Corporation Limited"
    ),
    "Date of Birth Change": (
        "BIRTH CERTIFICATE\n"
        "This is to certify that RAVI KUMAR SHARMA\n"
        "was born on 05-07-1989\n"
        "at Government Hospital, Patna, Bihar.\n"
        "Registration No: BIH/PTN/1989/00321\n"
        "Municipal Corporation of Patna"
    ),
    "Contact / Email Change": (
        "CUSTOMER CONSENT FORM\n"
        "I hereby authorize ABC Bank to update my contact details.\n"
        "New Mobile: 9876543210\n"
        "New Email: ravi.sharma@example.com\n"
        "I agree to the terms and conditions.\n"
        "Signature: [SIGNED]\n"
        "Date: 24-Apr-2024"
    ),
}


# ── Image quality estimator ───────────────────────────────────────────────────

def _image_quality_score(image: Image.Image) -> float:
    """
    Estimate readability (0-1) using pixel variance as a proxy for contrast.
    Higher variance = better quality / more readable document.
    """
    try:
        import cv2
        img_array = np.array(image.convert("L"))
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        return round(float(min(laplacian_var / 500.0, 1.0)), 4)
    except Exception:
        arr = np.array(image.convert("L"), dtype=float)
        return round(float(min(arr.var() / 5000.0, 1.0)), 4)


def _mock_ocr(change_type: str) -> str:
    return _MOCK_OCR_TEMPLATES.get(
        change_type,
        "DOCUMENT TEXT SAMPLE\nCustomer signature present."
    )


# ── OCR.space API caller ──────────────────────────────────────────────────────

def _call_ocr_space(file_bytes: bytes, filename: str = "document.png") -> str:
    """
    Send image bytes to OCR.space and return the parsed text.

    Raises:
        RuntimeError  – on HTTP error, API error flag, or empty result.
    """
    if not OCR_SPACE_API_KEY:
        raise RuntimeError("OCR_SPACE_API_KEY is not set in .env")

    # Encode as base64 data-URI so we can POST without multipart issues
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    # Detect mime type from first bytes
    if file_bytes[:4] == b"\x89PNG":
        mime = "image/png"
    elif file_bytes[:2] in (b"\xff\xd8", b"BM"):
        mime = "image/jpeg"
    else:
        mime = "image/png"  # safe fallback

    payload = {
        "apikey":          OCR_SPACE_API_KEY,
        "base64Image":     f"data:{mime};base64,{b64}",
        "language":        "eng",
        "isOverlayRequired": False,
        "detectOrientation": True,
        "scale":           True,
        "OCREngine":       2,           # Engine 2 is more accurate for printed text
    }

    logger.info("Calling OCR.space API (engine=2, mime=%s, bytes=%d)", mime, len(file_bytes))
    resp = requests.post(OCR_SPACE_URL, data=payload, timeout=30)
    resp.raise_for_status()

    data = resp.json()

    if data.get("IsErroredOnProcessing"):
        error_msg = data.get("ErrorMessage", ["Unknown API error"])
        if isinstance(error_msg, list):
            error_msg = "; ".join(error_msg)
        raise RuntimeError(f"OCR.space error: {error_msg}")

    parsed_results = data.get("ParsedResults", [])
    if not parsed_results:
        raise RuntimeError("OCR.space returned no parsed results.")

    text = "\n".join(
        r.get("ParsedText", "") for r in parsed_results
    ).strip()

    if not text:
        raise RuntimeError("OCR.space returned empty text.")

    logger.info("OCR.space success: %d chars extracted.", len(text))
    return text


# ── Public entry point ────────────────────────────────────────────────────────

def extract_text(
    file_bytes: bytes,
    change_type: str,
    use_mock: bool = False,         # explicit override from caller
) -> dict:
    """
    Main OCR entry point. Returns the standard output contract dict.

    Priority:
        1. USE_MOCK_OCR env flag  (or use_mock argument) → mock
        2. USE_OCR_API env flag   → OCR.space API
        3. API failure            → mock fallback
    """
    result: dict = {
        "text":        "",
        "ocr_quality": 0.7,
        "used_mock":   False,
        "source":      "",
        "error":       None,
    }

    # ── Compute image quality score (always, if image loads) ─────────────────
    try:
        image = Image.open(io.BytesIO(file_bytes))
        result["ocr_quality"] = _image_quality_score(image)
    except Exception as exc:
        logger.warning("Could not load image for quality check: %s", exc)
        result["ocr_quality"] = 0.5   # neutral default

    # ── Route 1: Force mock ──────────────────────────────────────────────────
    if use_mock or USE_MOCK_OCR:
        result["text"]      = _mock_ocr(change_type)
        result["used_mock"] = True
        result["source"]    = "mock"
        logger.info("Mock OCR used (forced). change_type=%s", change_type)
        return result

    # ── Route 2: OCR.space API ───────────────────────────────────────────────
    if USE_OCR_API:
        try:
            result["text"]      = _call_ocr_space(file_bytes)
            result["used_mock"] = False
            result["source"]    = "ocr_space"
            return result
        except Exception as exc:
            logger.error("OCR.space API failed: %s — using mock fallback.", exc)
            result["error"]     = str(exc)
            result["text"]      = _mock_ocr(change_type)
            result["used_mock"] = True
            result["source"]    = "fallback"
            return result

    # ── Route 3: No API configured → mock ────────────────────────────────────
    logger.info("USE_OCR_API=false and no local OCR; using mock. change_type=%s", change_type)
    result["text"]      = _mock_ocr(change_type)
    result["used_mock"] = True
    result["source"]    = "mock"
    return result
