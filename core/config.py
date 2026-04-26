"""
Global configuration and constants for the IASW system.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── Environment Flags ────────────────────────────────────────────────────────
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
USE_MOCK_LLM      = os.getenv("USE_MOCK_LLM", "true").lower() == "true" or not GEMINI_API_KEY
USE_MOCK_OCR      = os.getenv("USE_MOCK_OCR", "false").lower() == "true"

# OCR.space API (replaces local pytesseract)
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "helloworld")   # free demo key
USE_OCR_API       = os.getenv("USE_OCR_API", "true").lower() == "true"

# ─── Encryption ───────────────────────────────────────────────────────────────
ENCRYPTION_KEY    = os.getenv("ENCRYPTION_KEY", "")


# ─── Database ─────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iasw.db")

# ─── Confidence Thresholds ────────────────────────────────────────────────────
CONFIDENCE_PASS_THRESHOLD = 0.75   # ≥ 75% → PASS
CONFIDENCE_FLAG_THRESHOLD = 0.50   # 50–74% → FLAG (still possible to approve)
# < 50% → FAIL

# ─── Confidence Score Weights ─────────────────────────────────────────────────
WEIGHTS = {
    "data_match":          0.35,
    "authenticity":        0.25,
    "semantic_similarity": 0.20,
    "ocr_quality":         0.10,
    "business_rules":      0.10,
}

# ─── Supported Change Types ───────────────────────────────────────────────────
CHANGE_TYPES = [
    "Legal Name Change",
    "Address Change",
    "Date of Birth Change",
    "Contact / Email Change",
]

# ─── Document type keywords for authenticity engine ───────────────────────────
DOCUMENT_KEYWORDS = {
    "Legal Name Change": {
        "marriage_certificate": ["marriage certificate", "husband", "wife", "spouse",
                                  "solemnized", "registrar", "married"],
        "gazette":             ["gazette", "government of india", "notification",
                                 "name changed", "formerly known"],
    },
    "Address Change": {
        "utility_bill":        ["electricity", "water", "gas", "bill", "consumer",
                                 "account number", "due date"],
        "lease_agreement":     ["lease", "landlord", "tenant", "rent", "agreement"],
        "govt_id":             ["government", "india", "voter", "aadhaar", "aadhar",
                                 "passport", "driving licence"],
    },
    "Date of Birth Change": {
        "birth_certificate":   ["birth certificate", "date of birth", "registration",
                                 "municipality", "born"],
        "pan_card":            ["permanent account number", "income tax", "pan"],
        "passport":            ["passport", "republic of india", "date of birth",
                                 "nationality"],
    },
    "Contact / Email Change": {
        "consent_form":        ["consent", "authorize", "i hereby", "signature",
                                 "agree", "change request"],
    },
}

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "iasw_templates"
