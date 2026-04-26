"""
Address Validator – OpenStreetMap Nominatim API
────────────────────────────────────────────────
Validates a given address string against the Nominatim geocoding API.
Returns a structured result: found (bool), display_name (str), confidence (float).

No API key needed (Nominatim is free). Must respect usage policy:
  – Include a User-Agent header
  – Max 1 request per second (for bulk; single requests are fine)
"""
import re
import logging
import requests

logger = logging.getLogger(__name__)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
HEADERS       = {"User-Agent": "IASW-BankingSystem/1.0 (contact@bank.example.com)"}
TIMEOUT       = 10  # seconds


def _clean_address(address: str) -> str:
    """Strip common banking abbreviations and normalize spaces."""
    address = re.sub(r"\bNo\.?\s*\d+\b", "", address, flags=re.IGNORECASE)
    address = re.sub(r"\s+", " ", address).strip(" ,")
    return address


def validate_address(address: str) -> dict:
    """
    Query Nominatim to check if the address resolves to a real location.

    Returns:
        {
            "found":        bool
            "display_name": str | None   – best-match address from OSM
            "lat":          float | None
            "lon":          float | None
            "confidence":   float        – 0.0–1.0
            "error":        str | None
        }
    """
    result = {
        "found":        False,
        "display_name": None,
        "lat":          None,
        "lon":          None,
        "confidence":   0.0,
        "error":        None,
    }

    clean = _clean_address(address)
    if not clean:
        result["error"] = "Address is empty after cleaning."
        return result

    try:
        params = {
            "q":              clean,
            "format":         "json",
            "addressdetails": 1,
            "limit":          1,
            "countrycodes":   "in",   # restrict to India for banking use-case
        }
        resp = requests.get(NOMINATIM_URL, params=params,
                            headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            result["error"] = "No results found for this address."
            logger.info("Nominatim: no match for '%s'", clean)
            return result

        top = data[0]
        importance = float(top.get("importance", 0.0))

        result["found"]        = True
        result["display_name"] = top.get("display_name", "")
        result["lat"]          = float(top.get("lat", 0.0))
        result["lon"]          = float(top.get("lon", 0.0))
        result["confidence"]   = round(min(importance, 1.0), 4)
        logger.info("Nominatim match: '%s' → %.4f importance", result["display_name"][:60], importance)

    except requests.Timeout:
        result["error"] = "Nominatim API timed out."
        logger.warning("Nominatim timed out for address: %s", clean)
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning("Nominatim error: %s", exc)

    return result
