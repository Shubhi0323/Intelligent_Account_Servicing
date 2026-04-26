# Intelligent Account Servicing Workflow (IASW) - Project Summary

## Executive Summary
The Intelligent Account Servicing Workflow (IASW) is an automated, AI-driven document validation and fraud detection system designed to streamline bank-related customer change requests. By orchestrating a pipeline of intelligent agents, the system drastically reduces manual review times for updates such as legal name, address, and date of birth changes. It leverages OCR for text extraction, vector databases for semantic template matching, heuristic algorithms for fraud detection, and Large Language Models (LLMs) to synthesize a comprehensive, risk-adjusted confidence score and professional summary, enabling Human-in-the-Loop (HITL) checkers to make faster, more accurate decisions.

## Problem Understanding & Scope
**The Problem:** Banks receive thousands of customer requests daily to update personal information. Currently, verifying the authenticity of the attached supporting documents (e.g., utility bills, government IDs) is a highly manual, error-prone, and time-consuming process. Checkers must verify that the document matches the requested change and hasn't been tampered with.

**The Scope:** 
- **In Scope:** Automated ingestion of change requests (Name, Address, DOB, Contact) and their supporting document images. Processing the document to extract text, checking for basic tampering/fraud, verifying that the extracted data logically matches the requested new value, and providing a synthesized summary and PASS/FLAG/FAIL recommendation to a human reviewer.
- **Out of Scope:** The system does not directly mutate the core banking database with the new values; it acts as a decision-support validation layer. Advanced deepfake image detection is not currently supported natively.

## Solution Architecture
The system is built as a modular Python backend orchestrated using **LangGraph**, which treats the validation steps as a directed graph of autonomous nodes:

1. **Input Validation:** Ensures required metadata (change type, old/new values, file bytes) is present.
2. **OCR Processing:** Uses the **OCR.space REST API** to extract text and assess text quality from the uploaded image.
3. **Parallel Analysis Layer:**
   - **Authenticity Check:** Evaluates document completeness, tampering heuristics, and structure.
   - **Semantic Similarity:** Uses **ChromaDB** to embed the OCR text and compare it against known, valid templates (e.g., standard formats for PAN cards or utility bills).
   - **Validation Agent:** Executes domain-specific business rules (e.g., cross-checking the provided "new value" against the OCR text).
   - **Fraud Detection:** Analyzes image-level heuristics (using `opencv-python`) and fuzzy matching to detect risk flags.
4. **Confidence Scoring:** Aggregates the results of the parallel analysis layer into a final, weighted confidence percentage.
5. **Summary Generation:** Passes the aggregated data to the AI Summary Agent.
6. **Database Persistence:** Saves the request, the sub-scores, and the final AI summary to a SQLite database (`iasw.db`) using SQLAlchemy.

## Agent Design & Prompt Engineering
The system utilizes a hybrid approach, combining deterministic, rule-based "agents" with a generative AI agent for the final synthesis.

**1. Specialized Node Agents:**
Rather than relying on a single monolithic LLM to do everything, IASW routes specific tasks to specialized nodes. For example, the `fraud_detector` agent focuses strictly on fuzzy text matching and image heuristics, while the `validation_agent` handles strict business logic (e.g., verifying address formats).

**2. Summary Generation Agent (Google Gemini):**
The final node acts as a synthesizer. It takes the deterministic outputs (scores, extracted values, flags) and uses **Gemini 1.5 Flash** to generate a human-readable report. 

**Prompt Engineering Strategy:**
The prompt uses a highly structured, context-injection approach to prevent hallucination. Instead of asking the LLM to analyze the raw document, the system passes pre-calculated facts:
```text
You are a bank compliance AI assistant. Write a concise, professional verification summary for a human checker.

Change Type    : {change_type}
Old Value      : {old_value}
New Value      : {new_value}
Confidence     : {pct}% [{status}]
Score Breakdown: {breakdown}
Findings       : {findings}

Write 3–5 sentences covering: what was verified, key findings, confidence explanation, and your recommendation (Approve/Flag/Reject). Be factual and professional.
```
*Design Choice:* By explicitly constraining the LLM to output 3-5 sentences based *only* on the provided `findings` and `breakdown`, we ensure the model acts as a reliable formatter rather than an unpredictable decision-maker. A local, rule-based mock summarizer is also included as a fallback if the API is unreachable.

## Assumptions, Constraints & Known Limitations
- **OCR Dependency:** The system relies on the external **OCR.space API**. Poor quality uploads (blurry, low lighting) will result in low OCR quality scores. If the API fails, it falls back to mock data.
- **Mock Fallbacks:** To ensure offline development capability, the system contains mock fallbacks for both OCR and the LLM. If deployed in production, these must be strictly disabled.
- **Fraud Detection Limits:** The current `fraud_detector.py` uses lightweight heuristics (e.g., basic OpenCV checks, mismatched names). It is not a substitute for enterprise-grade cryptographic document verification or deepfake detection tools.
- **Processing Time:** Running external APIs (OCR.space, Gemini) and local OpenCV checks sequentially per request takes a few seconds. The system is designed for asynchronous background processing rather than real-time synchronous UI blocking.
