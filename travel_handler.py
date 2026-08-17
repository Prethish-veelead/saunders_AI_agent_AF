"""
Travel request handler.
------------------------

Implements the travel_v1 schema as single-shot extraction rather than a
field-by-field form walk. No approver field — a request is "extracted" as a
flat list of legs:

  [ { "destinationCountry": "...", "startDate": "YYYY-MM-DD",
      "endDate": "YYYY-MM-DD", "reason": "..." }, ... ]

Design decisions (per current requirements):
  1. One Azure OpenAI call extracts everything the message states — including
     multiple destinations in a single message — rather than asking one
     field at a time.
  2. Date normalization happens in the extraction prompt itself (the model
     converts "26 Aug 2026" / "26-Aug-2026" / etc. to ISO YYYY-MM-DD).
  3. Required fields are validated in code, never just trusted from the model.
  4. No server-side session. State is carried via the request/response
     round-trip: the caller resends the previous response as
     `pending_request` along with the answer to the single question that was
     asked. Follow-up answers are applied directly to the known missing
     field — no second extraction call needed, which is both cheaper and
     more reliable than re-running full extraction on a fragment of text.
  5. Date sanity checks happen in code: endDate not before startDate,
     startDate not before today.
"""

import json
import logging
import os
import time
from datetime import date
from typing import Any, Optional

import requests
from dateutil import parser as date_parser

logger = logging.getLogger("travel_handler")

AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "1.5"))


class TravelExtractionError(Exception):
    """Raised when the extraction call fails or returns unusable output."""


def _with_retries(fn, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.RequestException, TravelExtractionError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning("Extraction call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                           attempt, MAX_RETRIES, exc, delay)
            time.sleep(delay)
    raise TravelExtractionError(f"Extraction failed after {MAX_RETRIES} attempts") from last_exc


# --------------------------------------------------------------------------
# Step 1: single-shot extraction (first message only)
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """You extract structured travel request data from a user's message.

Respond ONLY with a JSON object matching this exact schema, nothing else:
{
  "legs": [
    {
      "destinationCountry": string or null,
      "startDate": string or null,
      "endDate": string or null,
      "reason": string or null
    }
  ]
}

Rules:
- Create one entry in "legs" for each distinct destination/trip mentioned in the message.
- Normalize every date to YYYY-MM-DD regardless of how it was written in the \
message (e.g. "26 Aug 2026" and "26-Aug-2026" both become "2026-08-26").
- If a field is not mentioned at all, use null. Never guess or invent a value.
- If no destinations are mentioned, return an empty "legs" array."""


def _extract_fields(text: str, config: dict[str, str]) -> list[dict[str, Any]]:
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise TravelExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raw_output = _with_retries(_call)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise TravelExtractionError(f"Extraction did not return valid JSON: {exc}") from exc

    legs = parsed.get("legs")
    if not isinstance(legs, list):
        legs = []
    normalized_legs = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        normalized_legs.append({
            "destinationCountry": leg.get("destinationCountry"),
            "startDate": leg.get("startDate"),
            "endDate": leg.get("endDate"),
            "reason": leg.get("reason"),
        })

    return normalized_legs


# --------------------------------------------------------------------------
# Step 2 & 5: validation in code, including date sanity checks
# --------------------------------------------------------------------------

def _validate(extracted: list[dict[str, Any]]) -> tuple[bool, Optional[str], Optional[str]]:
    """Returns (is_complete, missing_field_path, follow_up_question).

    Checks each leg in turn and returns the FIRST problem found — we ask
    one thing at a time. No approver field — travel requests only need
    destination, dates, and reason per leg.
    """
    if not extracted:
        return False, "0.destinationCountry", "Where are you traveling to?"

    today = date.today()
    for i, leg in enumerate(extracted):
        country = leg.get("destinationCountry")
        if not country:
            return False, f"{i}.destinationCountry", f"What's the destination country for trip #{i + 1}?"

        start_raw = leg.get("startDate")
        end_raw = leg.get("endDate")
        if not start_raw:
            return False, f"{i}.startDate", f"What's the start date for the trip to {country}?"
        if not end_raw:
            return False, f"{i}.endDate", f"What's the end date for the trip to {country}?"

        try:
            start_dt = date.fromisoformat(start_raw)
        except (ValueError, TypeError):
            return False, f"{i}.startDate", (
                f"I couldn't understand the start date for the trip to {country} — "
                f"could you confirm it?"
            )
        try:
            end_dt = date.fromisoformat(end_raw)
        except (ValueError, TypeError):
            return False, f"{i}.endDate", (
                f"I couldn't understand the end date for the trip to {country} — "
                f"could you confirm it?"
            )

        if start_dt < today:
            return False, f"{i}.startDate", (
                f"The start date for the trip to {country} is in the past — "
                f"could you confirm the correct start date?"
            )
        if end_dt < start_dt:
            return False, f"{i}.endDate", (
                f"The end date for the trip to {country} is before the start date — "
                f"could you confirm the correct end date?"
            )

        reason = leg.get("reason")
        if not reason:
            return False, f"{i}.reason", f"What's the reason for the trip to {country}?"

    return True, None, None


# --------------------------------------------------------------------------
# Step 4: stateless follow-up — apply one answer directly, no re-extraction
# --------------------------------------------------------------------------

def _apply_answer(
    extracted: list[dict[str, Any]], missing_field_path: str, answer_text: str
) -> list[dict[str, Any]]:
    answer_text = answer_text.strip()
    idx_str, field = missing_field_path.split(".")
    idx = int(idx_str)

    while len(extracted) <= idx:
        extracted.append({"destinationCountry": None, "startDate": None, "endDate": None, "reason": None})

    if field in ("startDate", "endDate"):
        try:
            extracted[idx][field] = date_parser.parse(answer_text, fuzzy=True).date().isoformat()
        except (ValueError, OverflowError):
            # Leave the raw text in place — validation will catch the bad
            # format and re-prompt with a clearer question.
            extracted[idx][field] = answer_text
    else:
        extracted[idx][field] = answer_text

    return extracted


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def process_travel_request(
    text: str, pending_request: Optional[dict[str, Any]], config: dict[str, str]
) -> dict[str, Any]:
    """Main entry point called from function_app.py.

    `extracted` is a flat list of legs — no approver field, no wrapper object.

    - If `pending_request` is provided (a previous incomplete result), the
      new `text` is treated as the answer to that single missing field —
      no extraction call happens, we merge directly.
    - Otherwise this is a fresh message: run full extraction.

    Returns either:
      {"status": "complete", "extracted": [...]}
    or:
      {"status": "incomplete", "extracted": [...], "missing_field_path": "...",
       "follow_up_question": "..."}
    """
    if pending_request and pending_request.get("missing_field_path"):
        extracted = pending_request.get("extracted") or []
        extracted = _apply_answer(extracted, pending_request["missing_field_path"], text)
    else:
        extracted = _extract_fields(text, config)

    is_complete, missing_field_path, follow_up_question = _validate(extracted)

    if is_complete:
        return {"status": "complete", "extracted": extracted}

    return {
        "status": "incomplete",
        "extracted": extracted,
        "missing_field_path": missing_field_path,
        "follow_up_question": follow_up_question,
    }
