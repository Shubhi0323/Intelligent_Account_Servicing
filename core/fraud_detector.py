"""
Fraud Detector
──────────────
Rule-based fraud detection engine using free heuristics.

Checks
──────
  1. Image Blur Detection     – OpenCV Laplacian variance
  2. Image Noise Detection    – Gaussian blur delta comparison
  3. Fuzzy Name Matching      – difflib SequenceMatcher
  4. Fuzzy Address Matching   – difflib (Address Change only)
  5. OCR Quality Check        – low confidence flag
  6. Semantic Similarity Check– low similarity flag
  7. Missing Fields Check     – field completeness from authenticity layers

Output
──────
  fraud_flags:   list[str]     e.g. ["name_mismatch", "blurry_document"]
  fraud_score:   float         0.0 (clean) → 1.0 (very suspicious)
  risk_level:    str           "LOW" | "MEDIUM" | "HIGH"
  fraud_details: list[str]     human-readable explanations
  image_checks:  dict          blur_variance, noise_ratio
"""

import io
import logging
from difflib import SequenceMatcher

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FLAG WEIGHTS — how much each flag contributes to fraud_score
# ═══════════════════════════════════════════════════════════════════════════════

_FLAG_WEIGHTS = {
    "blurry_document":        0.20,
    "noisy_image":            0.15,
    "name_mismatch":          0.25,
    "address_mismatch":       0.20,
    "missing_fields":         0.15,
    "low_ocr_quality":        0.15,
    "low_semantic_similarity": 0.15,
}

# Risk thresholds
_RISK_LOW    = 0.25
_RISK_MEDIUM = 0.55


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Image Blur Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_blur(file_bytes: bytes) -> tuple[float, bool, str]:
    """
    Compute Laplacian variance of the image.
    Low variance = blurry image = potential fraud (obscured details).

    Returns: (blur_variance, is_blurry, detail_str)
    """
    try:
        import cv2
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
        img_array = np.array(image)
        laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
        is_blurry = laplacian_var < 50.0
        detail = (
            f"🔴 Image is very blurry (Laplacian variance: {laplacian_var:.1f} < 50). "
            f"Possible attempt to obscure document details."
            if is_blurry else
            f"✅ Image sharpness OK (Laplacian variance: {laplacian_var:.1f})."
        )
        return round(laplacian_var, 2), is_blurry, detail
    except Exception as exc:
        logger.warning("Blur detection failed: %s", exc)
        return 0.0, False, f"⚠️ Blur detection unavailable: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Image Noise Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _check_noise(file_bytes: bytes) -> tuple[float, bool, str]:
    """
    Compare original image with a Gaussian-blurred version.
    High mean absolute difference = noisy image = potential tampering.

    Returns: (noise_ratio, is_noisy, detail_str)
    """
    try:
        import cv2
        image = Image.open(io.BytesIO(file_bytes)).convert("L")
        img_array = np.array(image, dtype=np.float64)
        blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
        diff = np.abs(img_array - blurred)
        noise_ratio = float(diff.mean()) / 255.0
        is_noisy = noise_ratio > 0.08
        detail = (
            f"🔴 High image noise detected (ratio: {noise_ratio:.4f} > 0.08). "
            f"Possible digital tampering or poor scan quality."
            if is_noisy else
            f"✅ Image noise level OK (ratio: {noise_ratio:.4f})."
        )
        return round(noise_ratio, 4), is_noisy, detail
    except Exception as exc:
        logger.warning("Noise detection failed: %s", exc)
        return 0.0, False, f"⚠️ Noise detection unavailable: {exc}"


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3: Fuzzy Name Matching
# ═══════════════════════════════════════════════════════════════════════════════

def _fuzzy_ratio(a: str, b: str) -> float:
    """Case-insensitive fuzzy similarity ratio (0–1)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _check_name_match(customer_name: str, ocr_text: str) -> tuple[float, bool, str]:
    """
    Check if customer name fuzzy-matches any segment of the OCR text.
    Uses a sliding window approach over the text.

    Returns: (best_ratio, is_mismatch, detail_str)
    """
    if not customer_name.strip():
        return 1.0, False, "⚠️ No customer name provided for matching."

    name_lower = customer_name.strip().lower()
    text_lower = ocr_text.lower()

    # Exact substring check first
    if name_lower in text_lower:
        return 1.0, False, f"✅ Customer name '{customer_name}' found exactly in document."

    # Fuzzy: check against each line and sliding windows
    best_ratio = 0.0
    name_len = len(name_lower)
    lines = text_lower.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Full line comparison
        ratio = _fuzzy_ratio(name_lower, line)
        best_ratio = max(best_ratio, ratio)
        # Sliding window over the line
        for i in range(max(1, len(line) - name_len + 1)):
            window = line[i:i + name_len + 5]  # slight slack
            ratio = _fuzzy_ratio(name_lower, window)
            best_ratio = max(best_ratio, ratio)

    is_mismatch = best_ratio < 0.6
    if is_mismatch:
        detail = (
            f"🔴 FRAUD FLAG: Customer name '{customer_name}' poorly matched in document "
            f"(best fuzzy ratio: {best_ratio:.2f} < 0.60). Identity cannot be confirmed."
        )
    elif best_ratio < 0.8:
        detail = (
            f"🟡 Customer name '{customer_name}' partially matched "
            f"(fuzzy ratio: {best_ratio:.2f}). Minor spelling variation detected."
        )
    else:
        detail = (
            f"✅ Customer name '{customer_name}' matched well "
            f"(fuzzy ratio: {best_ratio:.2f})."
        )

    return round(best_ratio, 4), is_mismatch, detail


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 4: Fuzzy Address Matching (Address Change only)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_address_match(new_value: str, ocr_text: str, change_type: str) -> tuple[float, bool, str]:
    """
    For Address Change: fuzzy-match the requested new address against OCR text.

    Returns: (best_ratio, is_mismatch, detail_str)
    """
    if change_type != "Address Change":
        return 1.0, False, ""

    if not new_value.strip():
        return 0.0, True, "🔴 No address value provided for matching."

    # Compare address tokens rather than full string
    import re
    addr_tokens = [t for t in re.split(r"[\s,/\-]+", new_value.lower()) if len(t) > 2]
    text_lower = ocr_text.lower()

    if not addr_tokens:
        return 0.0, True, "🔴 Address has no meaningful tokens to match."

    hits = sum(1 for t in addr_tokens if t in text_lower)
    ratio = hits / len(addr_tokens)

    # Also do full fuzzy match
    full_ratio = _fuzzy_ratio(new_value, ocr_text[:500])
    best = max(ratio, full_ratio)

    is_mismatch = best < 0.4
    if is_mismatch:
        detail = (
            f"🔴 FRAUD FLAG: New address poorly matched in document "
            f"(token overlap: {ratio:.2f}, fuzzy: {full_ratio:.2f}). "
            f"Document may not correspond to claimed address."
        )
    else:
        detail = (
            f"✅ Address matched in document "
            f"(token overlap: {ratio:.2f}, fuzzy: {full_ratio:.2f})."
        )

    return round(best, 4), is_mismatch, detail


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 5–7: Score-Based Flags
# ═══════════════════════════════════════════════════════════════════════════════

def _check_ocr_quality(ocr_quality: float) -> tuple[bool, str]:
    if ocr_quality < 0.3:
        return True, f"🔴 FRAUD FLAG: Very low OCR quality ({ocr_quality:.2f} < 0.30). Document may be illegible or corrupted."
    return False, f"✅ OCR quality acceptable ({ocr_quality:.2f})."


def _check_semantic(semantic_score: float) -> tuple[bool, str]:
    if semantic_score < 0.5:
        return True, f"🔴 FRAUD FLAG: Document content does not match expected template (semantic: {semantic_score:.2f} < 0.50). Wrong document type suspected."
    return False, f"✅ Document type semantically matches expected template ({semantic_score:.2f})."


def _check_missing_fields(authenticity_layers: dict) -> tuple[bool, str]:
    completeness = authenticity_layers.get("completeness", 1.0)
    if completeness < 0.5:
        return True, f"🔴 FRAUD FLAG: Document missing critical fields (completeness: {completeness:.2f} < 0.50). Possible fabricated or incomplete document."
    return False, f"✅ Document field completeness OK ({completeness:.2f})."


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fraud(
    file_bytes:          bytes,
    ocr_text:            str,
    ocr_quality:         float,
    semantic_score:      float,
    authenticity_layers: dict,
    change_type:         str,
    old_value:           str,
    new_value:           str,
    customer_name:       str,
) -> dict:
    """
    Run all fraud detection checks and return a structured result.

    Returns:
        {
            "fraud_flags":   list[str],
            "fraud_score":   float,        # 0.0–1.0
            "risk_level":    str,          # "LOW" | "MEDIUM" | "HIGH"
            "fraud_details": list[str],
            "image_checks":  dict,         # blur_variance, noise_ratio
        }
    """
    flags   = []
    details = []

    # ── 1. Blur detection ─────────────────────────────────────────────────────
    blur_var, is_blurry, blur_detail = _check_blur(file_bytes)
    if is_blurry:
        flags.append("blurry_document")
    details.append(blur_detail)

    # ── 2. Noise detection ────────────────────────────────────────────────────
    noise_ratio, is_noisy, noise_detail = _check_noise(file_bytes)
    if is_noisy:
        flags.append("noisy_image")
    details.append(noise_detail)

    # ── 3. Fuzzy name matching ────────────────────────────────────────────────
    name_ratio, name_mismatch, name_detail = _check_name_match(customer_name, ocr_text)
    if name_mismatch:
        flags.append("name_mismatch")
    details.append(name_detail)

    # ── 4. Fuzzy address matching ─────────────────────────────────────────────
    addr_ratio, addr_mismatch, addr_detail = _check_address_match(new_value, ocr_text, change_type)
    if addr_mismatch:
        flags.append("address_mismatch")
    if addr_detail:
        details.append(addr_detail)

    # ── 5. OCR quality check ──────────────────────────────────────────────────
    ocr_flag, ocr_detail = _check_ocr_quality(ocr_quality)
    if ocr_flag:
        flags.append("low_ocr_quality")
    details.append(ocr_detail)

    # ── 6. Semantic similarity check ──────────────────────────────────────────
    sem_flag, sem_detail = _check_semantic(semantic_score)
    if sem_flag:
        flags.append("low_semantic_similarity")
    details.append(sem_detail)

    # ── 7. Missing fields check ───────────────────────────────────────────────
    field_flag, field_detail = _check_missing_fields(authenticity_layers)
    if field_flag:
        flags.append("missing_fields")
    details.append(field_detail)

    # ── Compute fraud score ───────────────────────────────────────────────────
    fraud_score = sum(_FLAG_WEIGHTS.get(f, 0.10) for f in flags)
    fraud_score = round(min(fraud_score, 1.0), 4)

    # ── Determine risk level ──────────────────────────────────────────────────
    if fraud_score >= _RISK_MEDIUM:
        risk_level = "HIGH"
    elif fraud_score >= _RISK_LOW:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    result = {
        "fraud_flags":   flags,
        "fraud_score":   fraud_score,
        "risk_level":    risk_level,
        "fraud_details": details,
        "image_checks": {
            "blur_variance": blur_var,
            "noise_ratio":   noise_ratio,
            "name_ratio":    name_ratio,
            "addr_ratio":    addr_ratio,
        },
    }

    logger.info("Fraud detection: score=%.4f risk=%s flags=%s", fraud_score, risk_level, flags)
    return result
