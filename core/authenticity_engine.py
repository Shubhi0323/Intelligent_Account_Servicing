"""
Document Authenticity Validation Engine
─────────────────────────────────────────
Multi-layer validation system. Each layer contributes a sub-score (0–1).
Final authenticity score = weighted average of all layers.

Layers:
  1. Template Validation     – keyword presence check
  2. Field Completeness      – required fields detected
  3. OCR Quality             – passed-in from document processor
  4. Data Consistency        – internal cross-field check
  5. Tampering Detection     – basic heuristic (unusual char patterns)
"""
import logging
import re
from core.config import DOCUMENT_KEYWORDS

logger = logging.getLogger(__name__)


# ─── Layer 1: Template Validation ────────────────────────────────────────────

def _template_score(text: str, change_type: str) -> float:
    """Check how many expected keywords are present in the extracted text."""
    text_lower = text.lower()
    doc_types  = DOCUMENT_KEYWORDS.get(change_type, {})
    if not doc_types:
        return 0.5  # unknown change type → neutral

    best_score = 0.0
    for keywords in doc_types.values():
        hits  = sum(1 for kw in keywords if kw in text_lower)
        score = hits / max(len(keywords), 1)
        best_score = max(best_score, score)
    return round(best_score, 4)


# ─── Layer 2: Field Completeness ─────────────────────────────────────────────

_REQUIRED_PATTERNS: dict[str, list[str]] = {
    "Legal Name Change": [
        r"certificate",
        r"name",
        r"date|day",
        r"sign|registrar|authority",
    ],
    "Address Change": [
        r"address|locality|sector|street|road|nagar|colony",
        r"pin|pincode|\d{6}",
        r"date|bill",
        r"name|consumer",
    ],
    "Date of Birth Change": [
        r"born|date of birth|dob",
        r"\d{2}[-/]\d{2}[-/]\d{4}|\d{4}",
        r"certificate|registration",
        r"name|holder",
    ],
    "Contact / Email Change": [
        r"consent|authorize|hereby",
        r"sign|signature",
        r"mobile|phone|email",
        r"date",
    ],
}


def _field_completeness_score(text: str, change_type: str) -> float:
    patterns = _REQUIRED_PATTERNS.get(change_type, [])
    if not patterns:
        return 0.5
    hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
    return round(hits / len(patterns), 4)


# ─── Layer 3: OCR Quality  (passed in) ───────────────────────────────────────
# already computed in document_processor.py; we just forward it here.


# ─── Layer 4: Data Consistency ───────────────────────────────────────────────

def _data_consistency_score(text: str, old_value: str, new_value: str) -> float:
    """
    Check if both old and new values are present in the extracted text.
    Score = 0.5 * (old found) + 0.5 * (new found)
    """
    text_lower     = text.lower()
    old_found      = old_value.lower() in text_lower
    new_found      = new_value.lower() in text_lower
    return round(0.5 * old_found + 0.5 * new_found, 4)


# ─── Layer 5: Tampering Detection (basic heuristic) ──────────────────────────

def _tampering_score(text: str) -> float:
    """
    Heuristic tamper check:
    - Excessive special characters can indicate copy-paste or digital forgery.
    - Very short text after image upload may indicate a blank/corrupt doc.
    - Returns 1.0 (clean) → 0.0 (very suspicious).
    """
    if len(text.strip()) < 30:
        return 0.2  # suspiciously short

    special_chars = re.findall(r"[^a-zA-Z0-9\s.,/\-:()'\"]", text)
    ratio = len(special_chars) / max(len(text), 1)

    if ratio > 0.15:
        return 0.3   # many odd chars – flag
    elif ratio > 0.08:
        return 0.6
    return 1.0


# ─── Aggregate ────────────────────────────────────────────────────────────────

def compute_authenticity(
    text: str,
    change_type: str,
    old_value: str,
    new_value: str,
    ocr_quality: float,
) -> dict:
    """
    Run all five layers and return an aggregate authenticity result.

    Returns:
        {
            "template":      float,
            "completeness":  float,
            "ocr_quality":   float,
            "consistency":   float,
            "tampering":     float,
            "score":         float   ← weighted aggregate (0–1)
        }
    """
    template     = _template_score(text, change_type)
    completeness = _field_completeness_score(text, change_type)
    consistency  = _data_consistency_score(text, old_value, new_value)
    tampering    = _tampering_score(text)

    # Weights within authenticity (internal)
    score = (
        template     * 0.30 +
        completeness * 0.25 +
        ocr_quality  * 0.20 +
        consistency  * 0.15 +
        tampering    * 0.10
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    layers = {
        "template":     template,
        "completeness": completeness,
        "ocr_quality":  ocr_quality,
        "consistency":  consistency,
        "tampering":    tampering,
        "score":        score,
    }
    logger.info("Authenticity layers: %s", layers)
    return layers
