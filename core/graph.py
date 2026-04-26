"""
IASW LangGraph Orchestration
─────────────────────────────
Defines the AI verification pipeline as a LangGraph StateGraph.

Nodes
─────
  1. input_validation        – sanity-check raw form inputs
  2. ocr_processing          – extract text from uploaded document image
  3. authenticity_check      – multi-layer document authenticity scoring
  4. semantic_similarity     – vector-DB template similarity scoring
  5. change_type_verification – change-type-specific field validation
  6. confidence_scoring      – aggregate weighted confidence score
  7. summary_generation      – human-readable AI summary
  8. save_to_db              – persist request to database

Graph flow
──────────
  input_validation
       │
  ocr_processing
       │
  ┌────┼────────────┐
  │    │             │
  authenticity  semantic  change_type
  │    │             │
  └────┼────────────┘
       │
  confidence_scoring
       │
  summary_generation
       │
  save_to_db
       │
    __end__
"""

import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from core.document_processor import extract_text
from core.authenticity_engine import compute_authenticity
from core.vector_store import compute_semantic_similarity
from core.validation_agent import validate
from core.fraud_detector import detect_fraud
from core.confidence_scorer import compute_confidence
from core.summary_generator import generate_summary
from core.database import save_request

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

class IASWState(TypedDict, total=False):
    """Shared state passed between all graph nodes."""

    # ── Inputs (set by caller / input_validation) ──
    customer_id:    str
    customer_name:  str
    change_type:    str
    old_value:      str
    new_value:      str
    file_bytes:     bytes
    created_by:     str

    # ── OCR outputs ──
    ocr_text:       str
    ocr_quality:    float
    ocr_source:     str
    ocr_used_mock:  bool

    # ── Scoring outputs ──
    authenticity_score:   float
    authenticity_layers:  dict
    semantic_score:       float
    data_match_score:     float
    business_rule_score:  float
    critical_mismatch:    bool
    validation_findings:  list[str]
    extracted_value:      str

    # ── Final outputs ──
    confidence:     float
    percentage:     float
    status:         str         # PASS | FLAG | FAIL
    breakdown:      dict
    breakdown_json: str
    explanation:    str
    summary:        str
    request_id:     str

    # ── Fraud detection outputs ──
    fraud_flags:    list[str]
    fraud_score:    float
    risk_level:     str         # LOW | MEDIUM | HIGH
    fraud_details:  list[str]
    image_checks:   dict

    # ── Error tracking ──
    error:          str | None


# ═══════════════════════════════════════════════════════════════════════════════
# NODE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def input_validation(state: IASWState) -> dict:
    """
    Node 1 — Validate that all required inputs are present.
    Raises ValueError on missing fields so the caller can display errors.
    """
    logger.info("── Node: input_validation ──")

    required = ["customer_id", "change_type", "old_value", "new_value", "file_bytes"]
    missing = [f for f in required if not state.get(f)]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    return {
        "customer_id":   state["customer_id"].strip(),
        "customer_name": state.get("customer_name", "").strip(),
        "change_type":   state["change_type"],
        "old_value":     state["old_value"].strip(),
        "new_value":     state["new_value"].strip(),
        "created_by":    state.get("created_by", ""),
    }


def ocr_processing(state: IASWState) -> dict:
    """
    Node 2 — Run OCR on the uploaded document image.
    Wraps core.document_processor.extract_text().
    """
    logger.info("── Node: ocr_processing ──")

    ocr = extract_text(state["file_bytes"], state["change_type"])

    return {
        "ocr_text":      ocr["text"],
        "ocr_quality":   ocr["ocr_quality"],
        "ocr_source":    ocr["source"],
        "ocr_used_mock": ocr["used_mock"],
    }


def authenticity_check(state: IASWState) -> dict:
    """
    Node 3 — Multi-layer document authenticity scoring.
    Wraps core.authenticity_engine.compute_authenticity().
    """
    logger.info("── Node: authenticity_check ──")

    auth = compute_authenticity(
        state["ocr_text"],
        state["change_type"],
        state["old_value"],
        state["new_value"],
        state["ocr_quality"],
    )

    return {
        "authenticity_score":  auth["score"],
        "authenticity_layers": auth,
    }


def semantic_similarity_node(state: IASWState) -> dict:
    """
    Node 4 — Semantic similarity scoring via ChromaDB / TF-IDF.
    Wraps core.vector_store.compute_semantic_similarity().
    """
    logger.info("── Node: semantic_similarity ──")

    score = compute_semantic_similarity(state["ocr_text"], state["change_type"])

    return {
        "semantic_score": score,
    }


def change_type_verification(state: IASWState) -> dict:
    """
    Node 5 — Change-type-specific field validation and business rules.
    Wraps core.validation_agent.validate().
    """
    logger.info("── Node: change_type_verification ──")

    val = validate(
        state["ocr_text"],
        state["change_type"],
        state["old_value"],
        state["new_value"],
        state.get("customer_name", ""),
    )

    return {
        "data_match_score":     val["data_match_score"],
        "business_rule_score":  val["business_rule_score"],
        "critical_mismatch":    val["critical_mismatch"],
        "validation_findings":  val["findings"],
        "extracted_value":      val["extracted_value"],
    }


def fraud_detection(state: IASWState) -> dict:
    """
    Node 5b — Fraud detection via rule-based heuristics.
    Wraps core.fraud_detector.detect_fraud().
    """
    logger.info("── Node: fraud_detection ──")

    fraud = detect_fraud(
        file_bytes          = state["file_bytes"],
        ocr_text            = state["ocr_text"],
        ocr_quality         = state["ocr_quality"],
        semantic_score      = state.get("semantic_score", 0.5),
        authenticity_layers = state.get("authenticity_layers", {}),
        change_type         = state["change_type"],
        old_value           = state["old_value"],
        new_value           = state["new_value"],
        customer_name       = state.get("customer_name", ""),
    )

    return {
        "fraud_flags":   fraud["fraud_flags"],
        "fraud_score":   fraud["fraud_score"],
        "risk_level":    fraud["risk_level"],
        "fraud_details": fraud["fraud_details"],
        "image_checks":  fraud["image_checks"],
    }


def confidence_scoring(state: IASWState) -> dict:
    """
    Node 6 — Aggregate all sub-scores into a final weighted confidence.
    Wraps core.confidence_scorer.compute_confidence().
    Incorporates fraud_score for risk-adjusted confidence.
    """
    logger.info("── Node: confidence_scoring ──")

    conf = compute_confidence(
        data_match_score    = state["data_match_score"],
        authenticity_score  = state["authenticity_score"],
        semantic_similarity = state["semantic_score"],
        ocr_quality         = state["ocr_quality"],
        business_rule_score = state["business_rule_score"],
        critical_mismatch   = state["critical_mismatch"],
        fraud_score         = state.get("fraud_score", 0.0),
    )

    return {
        "confidence":     conf["confidence"],
        "percentage":     conf["percentage"],
        "status":         conf["status"],
        "breakdown":      conf["breakdown"],
        "breakdown_json": conf["breakdown_json"],
        "explanation":    conf["explanation"],
    }


def summary_generation(state: IASWState) -> dict:
    """
    Node 7 — Generate a human-readable verification summary.
    Wraps core.summary_generator.generate_summary().
    """
    logger.info("── Node: summary_generation ──")

    summary = generate_summary(
        state["change_type"],
        state["old_value"],
        state["new_value"],
        state["confidence"],
        state["status"],
        state["validation_findings"],
        state["breakdown"],
    )

    return {
        "summary": summary,
    }


def save_to_db(state: IASWState) -> dict:
    """
    Node 8 — Persist the completed request to the database.
    Wraps core.database.save_request().
    """
    logger.info("── Node: save_to_db ──")

    request_id = save_request(
        customer_id      = state["customer_id"],
        change_type      = state["change_type"],
        old_value        = state["old_value"],
        new_value        = state["new_value"],
        extracted_value  = state["extracted_value"],
        confidence_score = state["confidence"],
        ai_summary       = state["summary"],
        score_breakdown  = state["breakdown_json"],
        ai_explanation   = state["explanation"],
        created_by       = state.get("created_by", ""),
    )

    return {
        "request_id": request_id,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Construct and compile the IASW LangGraph StateGraph.

    Flow:
        input_validation
              │
        ocr_processing
              │
        ┌─────┼──────────────┐
        │     │              │
        auth  semantic  change_type
        │     │              │
        └─────┼──────────────┘
              │
        confidence_scoring
              │
        summary_generation
              │
        save_to_db
              │
           __end__
    """
    graph = StateGraph(IASWState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    graph.add_node("input_validation",         input_validation)
    graph.add_node("ocr_processing",           ocr_processing)
    graph.add_node("authenticity_check",        authenticity_check)
    graph.add_node("semantic_similarity",       semantic_similarity_node)
    graph.add_node("change_type_verification",  change_type_verification)
    graph.add_node("fraud_detection",           fraud_detection)
    graph.add_node("confidence_scoring",        confidence_scoring)
    graph.add_node("summary_generation",        summary_generation)
    graph.add_node("save_to_db",               save_to_db)

    # ── Define edges ──────────────────────────────────────────────────────────

    # Entry point
    graph.set_entry_point("input_validation")

    # Sequential: input → OCR
    graph.add_edge("input_validation", "ocr_processing")

    # Fan-out: OCR → four parallel scoring nodes
    graph.add_edge("ocr_processing", "authenticity_check")
    graph.add_edge("ocr_processing", "semantic_similarity")
    graph.add_edge("ocr_processing", "change_type_verification")
    graph.add_edge("ocr_processing", "fraud_detection")

    # Fan-in: all four → confidence scoring
    graph.add_edge("authenticity_check",        "confidence_scoring")
    graph.add_edge("semantic_similarity",       "confidence_scoring")
    graph.add_edge("change_type_verification",  "confidence_scoring")
    graph.add_edge("fraud_detection",           "confidence_scoring")

    # Sequential: scoring → summary → save → end
    graph.add_edge("confidence_scoring",  "summary_generation")
    graph.add_edge("summary_generation",  "save_to_db")
    graph.add_edge("save_to_db",          END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

# Compile graph once at module load
_compiled_graph = None


def _get_graph():
    """Lazy-initialise the compiled graph (avoids import-time side-effects)."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(initial_state: dict) -> dict:
    """
    Execute the full IASW verification pipeline.

    Args:
        initial_state: dict with keys:
            - customer_id   (str)
            - customer_name (str)
            - change_type   (str)
            - old_value     (str)
            - new_value     (str)
            - file_bytes    (bytes)
            - created_by    (str, optional)

    Returns:
        Final state dict with all scoring, summary, and request_id fields.
    """
    logger.info("═══ IASW LangGraph Pipeline — START ═══")

    graph = _get_graph()
    final_state = graph.invoke(initial_state)

    logger.info("═══ IASW LangGraph Pipeline — END (status=%s, confidence=%.2f%%) ═══",
                final_state.get("status"), final_state.get("percentage", 0))

    return final_state
