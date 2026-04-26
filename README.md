# Intelligent Account Servicing Workflow (IASW)

A comprehensive, automated document validation and fraud detection engine built for processing bank-related customer change requests. 

## Overview

The IASW system automates the verification of customer requests (e.g., legal name changes, address changes, date of birth updates) by analyzing submitted documents. It orchestrates a series of validation steps—including OCR, authenticity scoring, semantic matching, change-type specific rule evaluation, and fraud detection—using **LangGraph**. The system calculates an aggregate confidence score, assesses fraud risk, and generates an AI summary to support Human-in-the-Loop (HITL) checker decisions.

## Key Features

- **LangGraph Orchestration:** A defined `StateGraph` that sequentially and in parallel processes inputs across multiple autonomous nodes (OCR, Authenticity, Similarity, Rules, Fraud Detection) before aggregating the final score.
- **Document Processing & OCR:** Utilizes the **OCR.space REST API** to extract text from document images, with a deterministic mock OCR fallback for offline environments or API failures. No local Tesseract installation is required.
- **Fraud Detection Engine:** A lightweight, rule-based fraud detection module that flags suspicious activities using image-level heuristics, fuzzy matching for identity verification, and cross-field consistency checks.
- **Authenticity Engine:** A multi-layered validation system calculating an authenticity score by checking template validation, field completeness, OCR quality, and tampering heuristics.
- **Semantic Similarity Store:** Uses ChromaDB with lightweight hash-based embeddings (or TF-IDF fallback) to compare extracted document text against known, valid templates.
- **Validation Agent:** Executes domain-specific business rules tailored for each change type to ensure structural and logical consistency (e.g., verifying old vs. new values).
- **AI Summary Generator:** Produces concise, actionable verification summaries using the **Google Gemini API**, falling back to a rule-based mock summarizer if offline.
- **Database Layer:** Tracks all verification requests, extracted data, AI confidence scores, and HITL checker decisions using a SQLite database (`iasw.db`) managed by SQLAlchemy.

## Supported Change Types

- **Legal Name Change:** Validates Marriage Certificates, Gazette Notifications, etc.
- **Address Change:** Validates Utility Bills, Lease Agreements, and Government IDs.
- **Date of Birth Change:** Validates Birth Certificates, PAN Cards, Passports, etc.
- **Contact / Email Change:** Validates Customer Consent Forms.

## Configuration

Configurations and constants are managed via `core/config.py` and the `.env` file. 

**Key Environment Variables (`.env`):**
- `GEMINI_API_KEY`: API key for generating summaries with Google Gemini.
- `USE_MOCK_LLM`: Set to `true` to use the offline mock summarizer instead of the Gemini API.
- `USE_MOCK_OCR`: Set to `true` to completely bypass the OCR API and use hardcoded mock templates for testing.
- `OCR_SPACE_API_KEY`: API key for OCR.space (defaults to 'helloworld' for free demo access).
- `USE_OCR_API`: Set to `true` to enable external OCR API calls.

**Confidence Thresholds:**
- `≥ 75%` → PASS
- `50–74%` → FLAG (Requires closer manual review)
- `< 50%` → FAIL

## Project Structure

```text
Bank/
├── .env                      # Environment variables
├── iasw.db                   # SQLite database (auto-generated)
├── README.md                 # Project documentation
├── test_pipeline.py          # Script to test the LangGraph pipeline
├── test_all_change_types.py  # Script testing various request scenarios
└── core/                     # Core engine modules
    ├── __init__.py
    ├── authenticity_engine.py  # Multi-layer authenticity scoring
    ├── confidence_scorer.py    # Aggregate confidence scorer
    ├── config.py               # Global configurations & thresholds
    ├── database.py             # SQLAlchemy DB schemas and queries
    ├── document_processor.py   # OCR and text extraction
    ├── fraud_detector.py       # Rule-based fraud detection engine
    ├── graph.py                # LangGraph pipeline orchestration
    ├── summary_generator.py    # AI and mock summary generation
    ├── validation_agent.py     # Change-type specific rule handlers
    └── vector_store.py         # Semantic similarity matching (ChromaDB)
```

## Setup & Installation

Ensure you have Python 3.9+ installed.

1. **Install Python Dependencies:**
   ```bash
   pip install pillow chromadb google-generativeai sqlalchemy python-dotenv numpy opencv-python langgraph requests
   ```
   *(Note: `opencv-python` is used for advanced OCR quality estimation and image-level fraud checks).*

2. **Configure Environment:**
   Update your `.env` file with your `GEMINI_API_KEY` and `OCR_SPACE_API_KEY`.

