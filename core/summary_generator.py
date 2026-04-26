"""
Summary Generator
──────────────────
Primary  : Google Gemini API (if GEMINI_API_KEY is set and USE_MOCK_LLM=false)
Fallback : Rule-based mock summariser (template-driven, no external calls)

The mock summariser produces realistic, human-readable summaries and
recommendations based on the confidence data – sufficient for end-to-end
prototype testing.
"""
import logging
from core.config import USE_MOCK_LLM, GEMINI_API_KEY

logger = logging.getLogger(__name__)


# ─── Mock Summariser ──────────────────────────────────────────────────────────

def _mock_summary(
    change_type:    str,
    old_value:      str,
    new_value:      str,
    confidence:     float,
    status:         str,
    findings:       list[str],
    breakdown:      dict,
) -> str:
    pct = round(confidence * 100, 1)
    recommendation = {
        "PASS": "✅ Recommend: APPROVE. Confidence is high.",
        "FLAG": "⚠️ Recommend: REVIEW CAREFULLY. Confidence is moderate – please verify manually.",
        "FAIL": "❌ Recommend: REJECT. Confidence is too low to proceed.",
    }.get(status, "Review manually.")

    findings_text = "\n".join(f"  • {f}" for f in findings[:5])

    summary = (
        f"AI Verification Summary — {change_type}\n"
        f"{'─' * 50}\n"
        f"Customer requested change from:\n"
        f"  '{old_value}'  →  '{new_value}'\n\n"
        f"Confidence Score : {pct}%  [{status}]\n\n"
        f"Score Breakdown:\n"
        f"  • Data Match          : {round(breakdown.get('data_match', 0) * 100, 1)}%\n"
        f"  • Authenticity        : {round(breakdown.get('authenticity', 0) * 100, 1)}%\n"
        f"  • Semantic Similarity : {round(breakdown.get('semantic_similarity', 0) * 100, 1)}%\n"
        f"  • OCR Quality         : {round(breakdown.get('ocr_quality', 0) * 100, 1)}%\n"
        f"  • Business Rules      : {round(breakdown.get('business_rules', 0) * 100, 1)}%\n\n"
        f"Key Findings:\n{findings_text}\n\n"
        f"{recommendation}"
    )
    return summary


# ─── Gemini Summariser (real LLM) ─────────────────────────────────────────────

def _gemini_summary(
    change_type: str,
    old_value:   str,
    new_value:   str,
    confidence:  float,
    status:      str,
    findings:    list[str],
    breakdown:   dict,
) -> str:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        pct = round(confidence * 100, 1)
        prompt = (
            f"You are a bank compliance AI assistant. Write a concise, professional "
            f"verification summary for a human checker.\n\n"
            f"Change Type    : {change_type}\n"
            f"Old Value      : {old_value}\n"
            f"New Value      : {new_value}\n"
            f"Confidence     : {pct}% [{status}]\n"
            f"Score Breakdown: {breakdown}\n"
            f"Findings       : {findings}\n\n"
            f"Write 3–5 sentences covering: what was verified, key findings, "
            f"confidence explanation, and your recommendation (Approve/Flag/Reject). "
            f"Be factual and professional."
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as exc:
        logger.warning("Gemini API failed (%s), falling back to mock summary.", exc)
        return _mock_summary(change_type, old_value, new_value,
                             confidence, status, findings, breakdown)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_summary(
    change_type: str,
    old_value:   str,
    new_value:   str,
    confidence:  float,
    status:      str,
    findings:    list[str],
    breakdown:   dict,
) -> str:
    """
    Generate a human-readable verification summary.
    Uses Gemini if configured, otherwise falls back to rule-based mock.
    """
    if USE_MOCK_LLM:
        logger.info("Using mock summary generator.")
        return _mock_summary(change_type, old_value, new_value,
                             confidence, status, findings, breakdown)

    logger.info("Using Gemini summary generator.")
    return _gemini_summary(change_type, old_value, new_value,
                           confidence, status, findings, breakdown)
