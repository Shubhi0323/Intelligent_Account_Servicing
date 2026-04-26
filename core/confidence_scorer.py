"""
Confidence Scorer
──────────────────
Combines all sub-scores into a final weighted confidence score.

Weights:
  data_match          30%
  authenticity        25%
  semantic_similarity 25%
  ocr_quality         10%
  business_rules      10%

Final status:
  PASS  → score ≥ 0.75
  FLAG  → 0.50 ≤ score < 0.75
  FAIL  → score < 0.50
"""
import json
import logging
from core.config import WEIGHTS, CONFIDENCE_PASS_THRESHOLD, CONFIDENCE_FLAG_THRESHOLD

logger = logging.getLogger(__name__)


def compute_confidence(
    data_match_score:     float,
    authenticity_score:   float,
    semantic_similarity:  float,
    ocr_quality:          float,
    business_rule_score:  float,
    critical_mismatch:    bool = False,
    fraud_score:          float = 0.0,
) -> dict:
    """
    Returns:
        {
            "confidence":  float (0–1),
            "percentage":  float (0–100),
            "status":      "PASS" | "FLAG" | "FAIL",
            "breakdown":   dict,
            "breakdown_json": str,
        }
    """
    # Determine fraud risk level
    if fraud_score >= 0.55:
        fraud_risk = "HIGH"
    elif fraud_score >= 0.25:
        fraud_risk = "MEDIUM"
    else:
        fraud_risk = "LOW"

    breakdown = {
        "data_match":          round(data_match_score, 4),
        "authenticity":        round(authenticity_score, 4),
        "semantic_similarity": round(semantic_similarity, 4),
        "ocr_quality":         round(ocr_quality, 4),
        "business_rules":      round(business_rule_score, 4),
        "fraud_risk":          fraud_risk,
    }

    confidence = (
        data_match_score     * WEIGHTS["data_match"]          +
        authenticity_score   * WEIGHTS["authenticity"]         +
        semantic_similarity  * WEIGHTS["semantic_similarity"]  +
        ocr_quality          * WEIGHTS["ocr_quality"]          +
        business_rule_score  * WEIGHTS["business_rules"]
    )
    
    if critical_mismatch:
        confidence *= 0.5  # Reduce confidence significantly

    # Apply fraud risk penalty
    if fraud_risk == "HIGH":
        confidence *= 0.70   # -30% for high fraud risk
    elif fraud_risk == "MEDIUM":
        confidence *= 0.85   # -15% for medium fraud risk
        
    confidence = round(min(max(confidence, 0.0), 1.0), 4)

    if confidence >= CONFIDENCE_PASS_THRESHOLD:
        if critical_mismatch or fraud_risk == "HIGH":
            status = "FLAG"
        else:
            status = "PASS"
    elif confidence >= CONFIDENCE_FLAG_THRESHOLD:
        status = "FLAG"
    else:
        status = "FAIL"

    # Generate Explainability Reason
    explanation_parts = []
    if status == "PASS":
        explanation_parts.append("Recommendation: **APPROVE**.")
    elif status == "FLAG":
        explanation_parts.append("Recommendation: **MANUAL REVIEW REQUIRED**.")
    else:
        explanation_parts.append("Recommendation: **REJECT**.")

    if data_match_score >= 0.8:
        explanation_parts.append("Strong data match found between requested values and document.")
    elif data_match_score >= 0.5:
        explanation_parts.append("Partial data match. Minor penalties applied due to missing secondary fields.")
    else:
        explanation_parts.append("Weak data match. Penalties applied due to missing critical fields.")
        
    if authenticity_score < 0.5:
        explanation_parts.append("Document authenticity is questionable (potential fraud or blurriness).")
        
    if business_rule_score < 1.0:
        explanation_parts.append("Failed one or more business logic rules (e.g., date formats, missing context).")
        
    if semantic_similarity >= 0.7:
        explanation_parts.append("Document type aligns well with the change request context.")

    # Fraud risk explanation
    if fraud_risk == "HIGH":
        explanation_parts.append("🔴 HIGH FRAUD RISK detected. Multiple suspicious signals found.")
    elif fraud_risk == "MEDIUM":
        explanation_parts.append("🟡 MEDIUM FRAUD RISK. Some suspicious signals require attention.")

    explanation = " ".join(explanation_parts)

    result = {
        "confidence":     confidence,
        "percentage":     round(confidence * 100, 2),
        "status":         status,
        "breakdown":      breakdown,
        "breakdown_json": json.dumps(breakdown),
        "explanation":    explanation,
    }
    logger.info("Confidence score=%.4f status=%s fraud_risk=%s", confidence, status, fraud_risk)
    return result

