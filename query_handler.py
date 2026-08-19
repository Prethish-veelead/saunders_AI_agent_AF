"""
Query/lookup request handler.
------------------------------

Handles "show me my ..." style requests: looking up an EXISTING travel
request or helpdesk ticket, as opposed to creating a new one. This module
only classifies which record type is being asked about and extracts
search filters from the message — it does not perform the actual lookup.
There's no persistence layer for either record type in this codebase yet,
so whatever consumes this response owns running the real search.

Design (mirrors travel_handler.py):
  1. One AI call extracts record_type + filters from the message.
  2. If record_type is missing/ambiguous, that's asked about first — the
     filter schema itself depends on which record type is being queried.
  3. If record_type is known but every filter is null, ask for at least
     one. Unlike travel request creation, filters aren't all required —
     just enough to narrow down a search.
  4. No server-side session — state travels via `previous_response`, the
     same stateless pattern as travel_request follow-ups.
"""

import json
import logging
import os
import time
from datetime import date
from typing import Any, Optional

import requests

logger = logging.getLogger("query_handler")

AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "1.5"))

RECORD_TYPES = ("travel_request", "helpdesk_ticket")

FILTER_KEYS = {
    "travel_request": ("destinationCountry", "startDate", "endDate", "dateRaised", "ticketId", "description"),
    "helpdesk_ticket": ("ticketId", "dateRaised", "subject", "status"),
}


class QueryExtractionError(Exception):
    """Raised when the extraction call fails or returns unusable output."""


def _with_retries(fn, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.RequestException, QueryExtractionError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Query extraction call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
    raise QueryExtractionError(f"Extraction failed after {MAX_RETRIES} attempts") from last_exc


def _empty_filters(record_type: str) -> dict[str, Any]:
    return {key: None for key in FILTER_KEYS[record_type]}


# --------------------------------------------------------------------------
# Step 1: single-shot extraction (fresh message)
# --------------------------------------------------------------------------

def _build_extraction_system_prompt() -> str:
    # Built per-call, not a static constant, so it always reflects the real
    # current date (see travel_handler.py for why this matters for dates).
    today = date.today().isoformat()
    return f"""You determine whether the user is asking to look up an EXISTING
record they (or their team) already submitted — a travel request or a
help-desk ticket — and extract search filters from their message.

This is a LOOKUP, not a request to create something new and not a new IT
support question. Only classify as one of the two record types below if
the message is clearly asking to see/find/check on something already
submitted (e.g. "show me...", "what's the status of...", "find my...").

Respond ONLY with a JSON object matching this exact schema, nothing else:
{{
  "record_type": "travel_request" | "helpdesk_ticket" | null,
  "filters": {{ ... }}
}}

Rules:
- Today's date is {today}.
- Set "record_type" to null if it's not clear which of the two they mean.
  In that case "filters" must be {{}} (empty object).
- If "record_type" is "travel_request", "filters" must have exactly these
  keys, each string or null: "destinationCountry", "startDate", "endDate",
  "dateRaised", "ticketId", "description".
- If "record_type" is "helpdesk_ticket", "filters" must have exactly these
  keys, each string or null: "ticketId", "dateRaised", "subject", "status".
- "dateRaised" is when the record was submitted, not a travel date — e.g.
  "raised on aug-4" means dateRaised, but "traveling from 26 Aug" on a
  travel_request lookup means startDate.
- "description" (travel) / "subject" (ticket) are free-text keyword search
  fields, not exact matches — fill them with whatever topic/keyword the
  user mentioned, if any.
- Normalize every date to YYYY-MM-DD. If a date is ambiguous and purely
  numeric (e.g. "7-9-2026" or "7/9/2026"), interpret it as DD-MM-YYYY (day
  before month) — never MM-DD-YYYY. If a date does not mention a year,
  assume the current year from today's date above.
- Use null for anything not mentioned. Never guess or invent a value."""


def _extract_query(text: str, config: dict[str, str]) -> dict[str, Any]:
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "system", "content": _build_extraction_system_prompt()},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise QueryExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raw_output = _with_retries(_call)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise QueryExtractionError(f"Extraction did not return valid JSON: {exc}") from exc

    record_type = parsed.get("record_type")
    if record_type not in RECORD_TYPES:
        return {"record_type": None, "filters": {}}

    raw_filters = parsed.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    filters = {key: raw_filters.get(key) or None for key in FILTER_KEYS[record_type]}

    return {"record_type": record_type, "filters": filters}


# --------------------------------------------------------------------------
# Step 2: validation — record_type first, then "at least one filter"
# --------------------------------------------------------------------------

def _validate(extracted: dict[str, Any]) -> tuple[bool, Optional[str], Optional[str]]:
    """Returns (is_complete, missing_field, follow_up_question).

    Unlike travel_request creation, filters aren't all required — a query
    just needs enough to narrow a search down, so any single non-null
    filter is sufficient once record_type is known.
    """
    record_type = extracted.get("record_type")
    if record_type not in RECORD_TYPES:
        return False, "record_type", "Is that a travel request or a helpdesk ticket?"

    filters = extracted.get("filters") or {}
    if not any(filters.get(key) for key in FILTER_KEYS[record_type]):
        if record_type == "travel_request":
            return False, "filters", "Which trip are you looking for — a destination, dates, or a request number?"
        return False, "filters", "Which ticket would you like to see — a ticket number, when it was raised, or what it's about?"

    return True, None, None


# --------------------------------------------------------------------------
# Step 3: stateless follow-up
# --------------------------------------------------------------------------

def _classify_record_type_answer(answer_text: str) -> Optional[str]:
    """Deterministic two-way match — no AI call needed for a binary choice
    like this, unlike filter extraction which can mention any combination
    of fields in free text.
    """
    lowered = answer_text.strip().lower()
    if "travel" in lowered or "trip" in lowered:
        return "travel_request"
    if "ticket" in lowered or "helpdesk" in lowered or "help desk" in lowered or "help-desk" in lowered:
        return "helpdesk_ticket"
    return None


def _extract_filters_from_answer(
    record_type: str, existing_filters: dict[str, Any], answer_text: str, config: dict[str, str]
) -> dict[str, Any]:
    """Scoped extraction for a free-text answer to "which filter(s) do you
    want to search by" — the user could mention any combination of the
    record type's filters in one reply, so this always runs through the AI
    rather than a deterministic parse (unlike a single known date field).
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    today = date.today().isoformat()
    keys = FILTER_KEYS[record_type]
    system_prompt = (
        f"The user is searching for an existing {record_type.replace('_', ' ')} and was "
        "asked to narrow it down. Extract any of the following filters their answer "
        f"actually states: {', '.join(keys)}.\n\n"
        f"Today's date is {today}.\n"
        f"Already-known filters (null means still unknown): {json.dumps(existing_filters)}\n\n"
        "Use null for anything not mentioned — never guess, invent, or repeat an "
        "already-filled value unless the user is clearly correcting it. Normalize "
        "any date to YYYY-MM-DD; if a date has no year, assume the current year; if "
        'ambiguous and purely numeric (e.g. "7-9-2026"), interpret it as DD-MM-YYYY.\n'
        "Respond ONLY with a JSON object with exactly these keys, nothing else:\n"
        + json.dumps({key: "string or null" for key in keys})
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_text},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise QueryExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raw_output = _with_retries(_call)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise QueryExtractionError(f"Follow-up extraction did not return valid JSON: {exc}") from exc

    return parsed if isinstance(parsed, dict) else {}


def process_query_request(
    text: str, pending_request: Optional[dict[str, Any]], config: dict[str, str]
) -> dict[str, Any]:
    """Main entry point called from function_app.py.

    - If `pending_request["missing_field"] == "record_type"`, `text` is the
      answer to "travel request or helpdesk ticket" — classified
      deterministically, no AI call needed.
    - If `pending_request["missing_field"] == "filters"`, `text` is a
      free-text answer that may mention any combination of filters — runs
      a scoped AI extraction rather than guessing which single field it
      maps to.
    - Otherwise this is a fresh message: run full extraction.

    Returns either:
      {"status": "complete", "query": {"record_type": ..., "filters": {...}}}
    or:
      {"status": "incomplete", "extracted": {...}, "missing_field": "...",
       "follow_up_question": "..."}
    """
    if pending_request and pending_request.get("missing_field") == "record_type":
        extracted = pending_request.get("extracted") or {"record_type": None, "filters": {}}
        record_type = _classify_record_type_answer(text)
        if record_type:
            extracted = {"record_type": record_type, "filters": _empty_filters(record_type)}
        # else: still unresolved — validate() below will re-ask.

    elif pending_request and pending_request.get("missing_field") == "filters":
        extracted = pending_request.get("extracted") or {"record_type": None, "filters": {}}
        record_type = extracted.get("record_type")
        filters = extracted.get("filters") or _empty_filters(record_type)
        updates = _extract_filters_from_answer(record_type, filters, text, config)
        for key in FILTER_KEYS.get(record_type, ()):
            value = updates.get(key)
            if value:
                filters[key] = value
        extracted["filters"] = filters

    else:
        extracted = _extract_query(text, config)

    is_complete, missing_field, follow_up_question = _validate(extracted)

    if is_complete:
        return {
            "status": "complete",
            "query": {"record_type": extracted["record_type"], "filters": extracted["filters"]},
        }

    return {
        "status": "incomplete",
        "extracted": extracted,
        "missing_field": missing_field,
        "follow_up_question": follow_up_question,
    }
