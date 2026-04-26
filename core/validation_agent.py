"""
Validation Agent
─────────────────
Change-type-specific rule engine.
Each handler checks domain rules and returns a data_match score (0–1)
plus a structured list of findings.

Handlers
────────
  1. Legal Name Change   – name presence + legal context
  2. Address Change      – keyword match + Nominatim geocoding
  3. Date of Birth Change – DOB extraction + format + logical check
  4. Contact/Email Change – consent + signature + format validation
"""
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _name_in_text(name: str, text: str) -> bool:
    return bool(name) and name.strip().lower() in text.lower()


def _extract_dob_strings(text: str) -> list[str]:
    """Return all date-like strings found in text."""
    patterns = [
        r"\b\d{2}[-/]\d{2}[-/]\d{4}\b",   # DD-MM-YYYY or DD/MM/YYYY
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",   # YYYY-MM-DD
        r"\b\d{2}\s+\w+\s+\d{4}\b",       # 05 July 1989
    ]
    found = []
    for p in patterns:
        found.extend(re.findall(p, text))
    return list(set(found))


def _parse_dob(dob_str: str) -> datetime | None:
    """Try multiple date formats; return datetime or None."""
    formats = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d",
               "%d %B %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(dob_str.strip(), fmt)
        except ValueError:
            continue
    return None


def _validate_dob_value(dob_str: str) -> tuple[bool, list[str]]:
    """
    Validate a DOB string. Returns (is_valid, findings_list).
    Checks: parseable format, not in the future, not before 1900.
    """
    issues = []
    dt = _parse_dob(dob_str)
    if dt is None:
        return False, [f"❌ DOB '{dob_str}' could not be parsed (expected DD-MM-YYYY)."]
    if dt > datetime.now():
        return False, [f"❌ DOB '{dob_str}' is a future date – invalid."]
    if dt.year < 1900:
        return False, [f"❌ DOB year {dt.year} is before 1900 – invalid."]
    age = (datetime.now() - dt).days // 365
    if age < 0 or age > 120:
        return False, [f"❌ Computed age {age} years is not plausible."]
    return True, [f"✅ DOB format valid. Age: ~{age} years."]


def _extract_address_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [l.strip() for l in lines if re.search(
        r"\d{6}|sector|block|nagar|street|road|colony|district|pin|flat|floor",
        l, re.IGNORECASE,
    )]


def _is_email(value: str) -> bool:
    return bool(re.match(r"^[\w\.\+\-]+@[\w\-]+\.\w{2,}$", value.strip()))


def _is_phone(value: str) -> bool:
    digits = re.sub(r"[\s\-\+\(\)]", "", value)
    return digits.isdigit() and 8 <= len(digits) <= 15


# ─── Handler 1: Legal Name Change ─────────────────────────────────────────────

def _validate_name_change(text: str, old_value: str, new_value: str, customer_name: str) -> dict:
    findings = []
    score = 0.20  # Base score

    old_found = _name_in_text(old_value, text)
    new_found = _name_in_text(new_value, text)

    critical_mismatch = False

    if new_found:
        score += 0.50
        findings.append(f"✅ New name '{new_value}' found in document.")
    else:
        score -= 0.20
        critical_mismatch = True
        findings.append(f"❌ CRITICAL: New name '{new_value}' NOT found in document.")

    if old_found:
        score += 0.10
        findings.append(f"✅ Old name '{old_value}' found in document.")
    else:
        findings.append(f"ℹ️ Old name '{old_value}' NOT found in document (no penalty applied).")

    legal_kws = ["marriage", "gazette", "court", "divorce", "solemnized",
                 "spouse", "husband", "wife", "registrar", "wedlock"]
    hits = [kw for kw in legal_kws if kw in text.lower()]
    if hits:
        score += 0.20
        findings.append(f"✅ Legal context keywords: {', '.join(hits[:3])}.")
    else:
        score -= 0.05
        findings.append("⚠️ No legal context keywords detected (minor penalty).")

    extracted_value = (
        f"Old: {'found' if old_found else 'missing'}, "
        f"New: {'found' if new_found else 'missing'}"
    )
    return {"score": round(max(0.0, min(score, 1.0)), 4), "findings": findings,
            "extracted_value": extracted_value, "critical_mismatch": critical_mismatch}


# ─── Handler 2: Address Change ────────────────────────────────────────────────

def _validate_address_change(text: str, old_value: str, new_value: str, customer_name: str) -> dict:
    from core.address_validator import validate_address
    findings = []
    score    = 0.20  # Base score
    critical_mismatch = False

    # ── Name check ────────────────────────────────────────────────────────────
    if customer_name and not _name_in_text(customer_name, text):
        critical_mismatch = True
        score -= 0.20
        findings.append(f"❌ CRITICAL: Customer name '{customer_name}' NOT found in document.")
    elif customer_name:
        score += 0.10
        findings.append(f"✅ Customer name '{customer_name}' found in document.")

    # ── 2a: Keyword overlap between new address and OCR text ─────────────────
    new_lower  = new_value.lower()
    new_words  = [w for w in re.split(r"[\s,/\-]+", new_lower) if len(w) > 3]
    hits       = [w for w in new_words if w in text.lower()]
    ratio      = len(hits) / max(len(new_words), 1)

    if ratio > 0.5:
        score += 0.30
        findings.append(f"✅ Address keywords in document: {', '.join(hits[:4])}.")
    else:
        score -= 0.20
        critical_mismatch = True
        findings.append(f"❌ CRITICAL: Low address keyword overlap in document ({int(ratio*100)}%).")

    # ── 2b: Pincode match ─────────────────────────────────────────────────────
    doc_pins = re.findall(r"\b\d{6}\b", text)
    req_pins = re.findall(r"\b\d{6}\b", new_value)
    if req_pins and req_pins[0] in doc_pins:
        score += 0.20
        findings.append(f"✅ Pincode {req_pins[0]} found in document.")
    elif req_pins:
        score -= 0.05
        findings.append(f"❌ Requested pincode {req_pins[0]} not found in document (minor penalty).")
    else:
        findings.append("⚠️ No pincode detected in requested address.")

    # ── 2c: Document type context ─────────────────────────────────────────────
    ctx_kws = ["bill", "consumer", "tenant", "lease", "landlord", "aadhaar", "voter", "utility"]
    if any(kw in text.lower() for kw in ctx_kws):
        score += 0.10
        findings.append("✅ Address proof document type confirmed (utility/lease/ID).")

    # ── 2d: Nominatim geocoding ───────────────────────────────────────────────
    nominatim_result = validate_address(new_value)
    if nominatim_result["found"]:
        geo_conf = nominatim_result["confidence"]
        geo_score = min(geo_conf * 0.10, 0.10)
        score += geo_score
        findings.append(
            f"✅ OpenStreetMap confirmed address: "
            f"{nominatim_result['display_name'][:80]}… "
            f"(confidence: {round(geo_conf*100)}%)"
        )
    elif nominatim_result.get("error"):
        findings.append(f"⚠️ OpenStreetMap validation unavailable: {nominatim_result['error']}")
    else:
        score -= 0.10
        findings.append("❌ Address not found in OpenStreetMap database (penalty applied).")

    addr_lines = _extract_address_lines(text)
    extracted_value = "; ".join(addr_lines[:2]) if addr_lines else "Address not clearly extracted"
    return {"score": round(max(0.0, min(score, 1.0)), 4), "findings": findings,
            "extracted_value": extracted_value, "critical_mismatch": critical_mismatch}


# ─── Handler 3: Date of Birth Change ─────────────────────────────────────────

def _validate_dob_change(text: str, old_value: str, new_value: str, customer_name: str) -> dict:
    findings = []
    score    = 0.20  # Base score
    critical_mismatch = False

    # ── Name check ────────────────────────────────────────────────────────────
    if customer_name and not _name_in_text(customer_name, text):
        critical_mismatch = True
        score -= 0.20
        findings.append(f"❌ CRITICAL: Customer name '{customer_name}' NOT found in document.")
    elif customer_name:
        score += 0.10
        findings.append(f"✅ Customer name '{customer_name}' found in document.")

    # ── 3a: Validate requested new DOB format and logic ───────────────────────
    new_valid, new_findings = _validate_dob_value(new_value)
    findings.extend(new_findings)
    if new_valid:
        score += 0.10
    else:
        # Hard failure if requested DOB itself is invalid
        findings.append("⚠️ Cannot verify an invalid requested DOB.")
        return {
            "score": 0.0, "findings": findings,
            "extracted_value": f"Requested DOB invalid: {new_value}",
            "critical_mismatch": True
        }

    # ── 3b: Validate old DOB format ───────────────────────────────────────────
    old_valid, old_findings = _validate_dob_value(old_value)
    if old_valid:
        score += 0.10
        findings.append(f"✅ Old DOB '{old_value}' format is valid.")
    else:
        score -= 0.05
        findings.append(f"⚠️ Old DOB '{old_value}' format issue (minor penalty).")

    # ── 3c: Extract DOBs from document ───────────────────────────────────────
    dobs = _extract_dob_strings(text)
    if dobs:
        findings.append(f"✅ Date(s) found in document: {', '.join(dobs[:3])}.")
        new_dt = _parse_dob(new_value)
        matched = False
        for d in dobs:
            doc_dt = _parse_dob(d)
            if doc_dt and new_dt and doc_dt.date() == new_dt.date():
                matched = True
                break
            # Fallback: substring match for differently formatted same date
            if new_value.strip() in d or d in new_value.strip():
                matched = True
                break
        if matched:
            score += 0.50
            findings.append(f"✅ Requested DOB '{new_value}' matches a date in document.")
        else:
            score -= 0.20
            critical_mismatch = True
            findings.append(f"❌ CRITICAL: Requested DOB '{new_value}' not matched in document (found: {', '.join(dobs[:2])}).")
    else:
        score -= 0.20
        critical_mismatch = True
        findings.append("❌ CRITICAL: No date pattern found in document.")

    # ── 3d: Document type context ─────────────────────────────────────────────
    ctx_kws = ["birth", "born", "registration", "municipal", "pan", "passport", "certificate"]
    ctx_hits = [k for k in ctx_kws if k in text.lower()]
    if ctx_hits:
        score += 0.10
        findings.append(f"✅ DOB document type confirmed: {', '.join(ctx_hits[:3])}.")

    extracted_value = f"Dates in document: {', '.join(dobs[:3])}" if dobs else "No DOB extracted"
    return {"score": round(max(0.0, min(score, 1.0)), 4), "findings": findings,
            "extracted_value": extracted_value, "critical_mismatch": critical_mismatch}


# ─── Handler 4: Contact / Email Change ───────────────────────────────────────

def _validate_contact_change(text: str, old_value: str, new_value: str, customer_name: str) -> dict:
    findings = []
    score    = 0.20  # Base score
    critical_mismatch = False

    # ── Name check ────────────────────────────────────────────────────────────
    if customer_name and not _name_in_text(customer_name, text):
        critical_mismatch = True
        score -= 0.20
        findings.append(f"❌ CRITICAL: Customer name '{customer_name}' NOT found in document.")
    elif customer_name:
        score += 0.10
        findings.append(f"✅ Customer name '{customer_name}' found in document.")

    # ── 4a: Format validation of the new value ────────────────────────────────
    new_clean = new_value.strip()
    if _is_email(new_clean):
        findings.append(f"✅ New value '{new_clean}' is a valid email format.")
        score += 0.10
        contact_type = "email"
    elif _is_phone(new_clean):
        findings.append(f"✅ New value '{new_clean}' is a valid phone number format.")
        score += 0.10
        contact_type = "phone"
    else:
        findings.append(f"⚠️ '{new_clean}' does not match standard email or phone format.")
        contact_type = "unknown"

    # ── 4b: Consent language ─────────────────────────────────────────────────
    consent_kws = ["consent", "authorize", "authorise", "i hereby", "agree",
                   "permission", "request", "change request"]
    consent_hits = [kw for kw in consent_kws if kw in text.lower()]
    if consent_hits:
        score += 0.25
        findings.append(f"✅ Consent language detected: '{consent_hits[0]}'.")
    else:
        score -= 0.10
        findings.append("❌ Consent language NOT detected in document (penalty applied).")

    # ── 4c: Signature detection ───────────────────────────────────────────────
    sig_kws = ["signature", "signed", "[signed]", "sign:", "sgd.", "authorised signatory"]
    sig_found = any(kw in text.lower() for kw in sig_kws)
    if sig_found:
        score += 0.15
        findings.append("✅ Signature indicator found in document.")
    else:
        score -= 0.05
        findings.append("⚠️ No signature indicator detected (minor penalty).")

    # ── 4d: New value presence in document ───────────────────────────────────
    if new_clean.lower() in text.lower():
        score += 0.20
        findings.append(f"✅ New {contact_type} '{new_clean}' explicitly found in form.")
    else:
        score -= 0.20
        critical_mismatch = True
        findings.append(f"❌ CRITICAL: New {contact_type} '{new_clean}' not explicitly found in form.")

    # ── 4e: Old value mentioned (for context) ────────────────────────────────
    old_clean = old_value.strip()
    if old_clean.lower() in text.lower():
        findings.append(f"✅ Old contact value '{old_clean}' also referenced in form.")

    extracted_value = f"New {contact_type}: {new_clean}"
    return {"score": round(max(0.0, min(score, 1.0)), 4), "findings": findings,
            "extracted_value": extracted_value, "critical_mismatch": critical_mismatch}


# ─── Router ───────────────────────────────────────────────────────────────────

_HANDLERS = {
    "Legal Name Change":      _validate_name_change,
    "Address Change":         _validate_address_change,
    "Date of Birth Change":   _validate_dob_change,
    "Contact / Email Change": _validate_contact_change,
}


def validate(
    text:        str,
    change_type: str,
    old_value:   str,
    new_value:   str,
    customer_name: str = "",
) -> dict:
    """
    Route to the correct handler and return a standardised result dict:
        {
            "data_match_score":    float,
            "findings":            list[str],
            "extracted_value":     str,
            "business_rule_score": float,
            "critical_mismatch":   bool,
        }
    """
    handler = _HANDLERS.get(change_type)
    if handler is None:
        logger.warning("No handler for change_type='%s'", change_type)
        return {
            "data_match_score":    0.5,
            "findings":            [f"⚠️ No validation rules defined for '{change_type}'."],
            "extracted_value":     "N/A",
            "business_rule_score": 0.5,
            "critical_mismatch":   False,
        }

    result = handler(text, old_value, new_value, customer_name)

    # Universal business rule: old value must differ from new value
    biz_ok = old_value.strip().lower() != new_value.strip().lower()
    biz_score = 1.0 if biz_ok else 0.0
    if not biz_ok:
        result["findings"].append("❌ Business Rule FAIL: Old value and new value are identical!")

    logger.info("Validation [%s]: score=%.4f | biz=%.1f | critical_mismatch=%s", 
                change_type, result["score"], biz_score, result["critical_mismatch"])
    return {
        "data_match_score":    result["score"],
        "findings":            result["findings"],
        "extracted_value":     result["extracted_value"],
        "business_rule_score": biz_score,
        "critical_mismatch":   result["critical_mismatch"],
    }
