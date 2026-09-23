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
  1. One AI call extracts record_type + filters from the message, plus
     "sort" ("recent"/"oldest"/null) and "limit" (integer/null) if the
     message expresses ordering or a count (e.g. "latest 2", "most
     recent", "oldest one").
  2. If record_type is missing/ambiguous, that's asked about first — the
     filter schema itself depends on which record type is being queried.
  3. If record_type is known but every filter is null AND no sort/limit
     was expressed, ask for at least one. Unlike travel request creation,
     filters aren't all required — just enough to narrow down a search
     (or a sort/limit is itself sufficient, e.g. "my latest travel
     request" has no filter at all but is a complete, specific query).
     An explicit "all"/"everything"/"any" answer is treated as a
     deliberate request for every record of that type, not a non-answer
     — it completes the query with empty filters instead of re-asking
     the same question forever.
  4. `travel_request` filters include "status" — null, or an array of one
     or more of "pending"/"approved"/"rejected"/"cancelled" (an array so
     a single message can ask for more than one status at once, e.g.
     "approved and rejected"). `helpdesk_ticket`'s existing "status"
     filter is unrelated — a free-text ticket status, not this fixed
     vocabulary.
  5. No server-side session — state travels via `previous_response`, the
     same stateless pattern as travel_request follow-ups.
"""

import json
import logging
import os
import re
import time
from datetime import date
from typing import Any, Optional

import requests

import ai_client

logger = logging.getLogger("query_handler")

AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "1.5"))

RECORD_TYPES = ("travel_request", "helpdesk_ticket")

FILTER_KEYS = {
    "travel_request": (
        "destinationCountry", "startDate", "endDate", "dateRaised",
        "ticketId", "description", "status",
    ),
    "helpdesk_ticket": ("ticketId", "dateRaised", "subject", "status"),
}

TRAVEL_STATUS_VALUES = ("pending", "approved", "rejected", "cancelled")
SORT_VALUES = ("recent", "oldest")


def _normalize_sort(value: Any) -> Optional[str]:
    return value if value in SORT_VALUES else None


def _normalize_limit(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit() and int(value.strip()) > 0:
        return int(value.strip())
    return None


def _normalize_travel_status(value: Any) -> Optional[list]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    seen: list = []
    for item in value:
        if isinstance(item, str) and item in TRAVEL_STATUS_VALUES and item not in seen:
            seen.append(item)
    return seen or None


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
help-desk ticket — and extract search filters, plus any sorting/limiting
they asked for, from their message.

This is a LOOKUP, not a request to create something new and not a new IT
support question. Only classify as one of the two record types below if
the message is clearly asking to see/find/check on something already
submitted (e.g. "show me...", "what's the status of...", "find my...").

Respond ONLY with a JSON object matching this exact schema, nothing else:
{{
  "record_type": "travel_request" | "helpdesk_ticket" | null,
  "filters": {{ ... }},
  "sort": "recent" | "oldest" | null,
  "limit": <integer> | null
}}

Rules:
- Today's date is {today}.
- Set "record_type" to null if it's not clear which of the two they mean.
  In that case "filters" must be {{}} (empty object) — but "sort"/"limit"
  should still be extracted if present, since ordering/count can be
  expressed independently of which record type it turns out to be. A
  DATE ALONE is NOT enough to decide record_type — both record types
  have dates (dateRaised on both, plus startDate/endDate on
  travel_request), so a date with no other distinguishing detail (no
  destination, no travel word, no IT/ticket word) must still result in
  null, not a guess: "show my request from 15 Oct 2026" and "check my
  request raised on 4 Aug" -> both null, even though each names a date,
  because nothing in either message says which record type it is. The
  reverse also holds: if the message names "travel request" or "helpdesk
  ticket" (or an obvious variant) ANYWHERE in it, record_type is NOT
  ambiguous, no matter how much other filler wording surrounds it —
  "first we can see travel request", "we can see travel request", and
  "see the travel request" must ALL resolve to "travel_request", the
  same as each other; extra words like "first" or "we can" before the
  actual topic never make it unclear.
- If "record_type" is "travel_request", "filters" must have exactly these
  keys: "destinationCountry", "startDate", "endDate", "dateRaised",
  "ticketId", "description" (each string or null), plus "status" — null,
  or an ARRAY of one or more of exactly these values: "pending",
  "approved", "rejected", "cancelled". Map synonyms onto this fixed set —
  "denied"/"declined" -> "rejected", "accepted" -> "approved", "awaiting
  approval"/"in review" -> "pending", "canceled"/"withdrawn" ->
  "cancelled". If the user names MORE THAN ONE status in the same
  message, include ALL of them in the array — "show me the latest 2
  approved and reject travel request ticket for cuba reason meeting
  client" -> status: ["approved", "rejected"]. Never invent a status
  that wasn't mentioned.
- If "record_type" is "helpdesk_ticket", "filters" must have exactly
  these keys, each string or null: "ticketId", "dateRaised", "subject",
  "status" — this "status" is a plain free-text ticket status like "open"
  or "closed" (whatever word the user used), NOT the fixed travel
  vocabulary above.
- "sort": "recent" when the user asks for the most recent/latest/newest
  record(s) ("latest", "most recent", "newest"). "oldest" when they ask
  for the earliest/first/oldest ("first one", "earliest", "oldest").
  null if no ordering is implied.
- "limit": the specific count the user asked for (e.g. "latest 2" -> 2,
  "top 3" -> 3, "show me 5 tickets" -> 5). If "sort" is set but no
  specific count is stated anywhere ("my latest request", "the most
  recent ticket" — no count word at all) -> default to 5, not 1 — the
  user didn't ask for exactly one, just for the most recent/oldest, so
  return a reasonable-size batch rather than assuming they want a single
  record. Only use exactly 1 if the user's own words say so (e.g. "just
  one", "a single request"). If neither "sort" nor an explicit count is
  present, leave "limit" null — don't assume any number just because a
  search is happening.
- "dateRaised" is when the record was submitted, not a travel date — e.g.
  "raised on aug-4" means dateRaised, but "traveling from 26 Aug" on a
  travel_request lookup means startDate.
- "description" (travel) / "subject" (ticket) are free-text keyword search
  fields, not exact matches — fill them with whatever topic/keyword the
  user mentioned, if any (e.g. "reason meeting client" -> "client
  meeting").
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
    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": _build_extraction_system_prompt()},
            {"role": "user", "content": text},
        ],
        max_tokens=400,
        response_format={"type": "json_object"},
    )

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
    sort = _normalize_sort(parsed.get("sort"))
    limit = _normalize_limit(parsed.get("limit"))

    if record_type not in RECORD_TYPES:
        return {"record_type": None, "filters": {}, "sort": sort, "limit": limit}

    raw_filters = parsed.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}
    filters = {}
    for key in FILTER_KEYS[record_type]:
        if record_type == "travel_request" and key == "status":
            filters[key] = _normalize_travel_status(raw_filters.get("status"))
        else:
            filters[key] = raw_filters.get(key) or None

    return {"record_type": record_type, "filters": filters, "sort": sort, "limit": limit}


# --------------------------------------------------------------------------
# Step 2: validation — record_type first, then "at least one filter"
# --------------------------------------------------------------------------

def _validate(extracted: dict[str, Any]) -> tuple[bool, Optional[str], Optional[str]]:
    """Returns (is_complete, missing_field, follow_up_question).

    Unlike travel_request creation, filters aren't all required — a query
    just needs enough to narrow a search down, so any single non-null
    filter is sufficient once record_type is known. A "sort" or "limit"
    alone (e.g. "my latest travel request" — no destination, no date, no
    status, just "latest") is also sufficient on its own, since it's a
    specific, narrowed request even without a data filter.
    """
    record_type = extracted.get("record_type")
    if record_type not in RECORD_TYPES:
        return False, "record_type", "Is that a travel request or a helpdesk ticket?"

    filters = extracted.get("filters") or {}
    has_filter = any(filters.get(key) for key in FILTER_KEYS[record_type])
    has_sort_or_limit = bool(extracted.get("sort") or extracted.get("limit"))
    if not has_filter and not has_sort_or_limit:
        if record_type == "travel_request":
            return False, "filters", "Which trip are you looking for — a destination, dates, status, or a request number?"
        return False, "filters", "Which ticket would you like to see — a ticket number, when it was raised, or what it's about?"

    return True, None, None


# --------------------------------------------------------------------------
# Step 3: stateless follow-up
# --------------------------------------------------------------------------

_SHOW_ALL_WORDS = {"all", "everything", "any", "anything"}


def _wants_all(text: str) -> bool:
    """Deterministic check: does the text explicitly ask for everything
    rather than narrowing down to something specific (e.g. "all", "show
    everything", "any of them")? Cheaper and more reliable than an AI call
    for something this unambiguous — and lets a deliberate "no filter"
    answer complete the query instead of looping the same follow-up
    question forever when nothing else gets extracted from it.
    """
    words = re.findall(r"[a-z']+", text.lower())
    return any(word in _SHOW_ALL_WORDS for word in words)


def _is_cancel_intent(text: str, config: dict[str, str]) -> bool:
    """Does this reply express a clear intent to CANCEL/ABORT/STOP the
    in-progress lookup entirely — not just decline to answer the current
    question, but end the whole search. Checked before anything else once
    a query is pending, mirroring travel_handler.py's same-purpose check.
    Fails open (treats as NOT a cancellation) if the check itself errors
    out — a broken check must never block a legitimate answer.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "The user has a lookup/search in progress (looking up an existing "
        "travel request or helpdesk ticket). Decide whether their reply "
        "expresses a clear intent to CANCEL/ABORT/STOP the search "
        "entirely — not merely decline to answer the current question, "
        'but actually end the whole process. Words like "cancel", '
        '"close", "exit", "quit", "abort", "never mind", "start over", '
        '"don\'t bother" or similar all count, tolerating typos and '
        "casual phrasing. A reply that attempts to answer the question, "
        "even vaguely, or that raises an unrelated topic without asking "
        "to stop, does NOT count.\n\n"
        'Respond ONLY with a JSON object: {"is_cancel": true or false}'
    )
    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=20,
        response_format={"type": "json_object"},
    )

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise QueryExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (QueryExtractionError, json.JSONDecodeError) as exc:
        logger.warning("Cancel-intent check failed, treating as not a cancellation: %s", exc)
        return False

    return parsed.get("is_cancel") is True


def _classify_record_type_answer(answer_text: str, config: dict[str, str]) -> Optional[str]:
    """AI-based disambiguation of the answer to "which record type" — a
    plain keyword match (the original approach here) has zero typo
    tolerance, unlike every other classification in this system (e.g.
    "helpdisk" wouldn't match "helpdesk"), and would silently re-ask the
    same question forever instead of resolving a typo'd answer.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        'The user was asked which kind of record they want to look up: a '
        "travel request or a helpdesk ticket. Decide which one their reply "
        "means, tolerating typos and casual phrasing. If the reply doesn't "
        "clearly indicate either one, say so.\n\n"
        'Respond ONLY with a JSON object: {"resolved": "travel_request" | '
        '"helpdesk_ticket" | "unclear"}'
    )
    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_text},
        ],
        max_tokens=20,
        response_format={"type": "json_object"},
    )

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise QueryExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (QueryExtractionError, json.JSONDecodeError) as exc:
        logger.warning("Record-type disambiguation call failed, treating as unclear: %s", exc)
        return None

    resolved = parsed.get("resolved")
    return resolved if resolved in RECORD_TYPES else None


def _extract_filters_from_answer(
    record_type: str,
    existing_filters: dict[str, Any],
    answer_text: str,
    config: dict[str, str],
    existing_sort: Optional[str] = None,
    existing_limit: Optional[int] = None,
) -> dict[str, Any]:
    """Scoped extraction for a free-text answer to "which filter(s) do you
    want to search by" — the user could mention any combination of the
    record type's filters (plus a sort/limit hint, e.g. "the latest one to
    cuba") in one reply, so this always runs through the AI rather than a
    deterministic parse (unlike a single known date field).
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    today = date.today().isoformat()
    keys = FILTER_KEYS[record_type]
    status_note = ""
    schema = {key: "string or null" for key in keys}
    if record_type == "travel_request":
        status_note = (
            ' "status" must be null, or an ARRAY of one or more of exactly '
            '"pending"/"approved"/"rejected"/"cancelled" (map synonyms onto '
            "this fixed set, and include every status named if more than "
            'one is mentioned, e.g. "approved and rejected" -> '
            '["approved", "rejected"]).'
        )
        schema["status"] = ["approved", "rejected"]
    schema["sort"] = "recent | oldest | null"
    schema["limit"] = "integer or null"
    system_prompt = (
        f"The user is searching for an existing {record_type.replace('_', ' ')} and was "
        "asked to narrow it down. Extract any of the following filters their answer "
        f"actually states: {', '.join(keys)}, plus top-level \"sort\" "
        '("recent"/"oldest"/null) and "limit" (integer/null) if the reply '
        'expresses ordering or a count (e.g. "the latest one", "top 2"). '
        'If "sort" is implied but no specific count is stated anywhere '
        '("the latest one", "the most recent" — no number), default '
        '"limit" to 5, not 1 — only use 1 if the user\'s own words say so '
        f'(e.g. "just one", "a single one").{status_note}\n\n'
        f"Today's date is {today}.\n"
        f"Already-known filters (null means still unknown): {json.dumps(existing_filters)}\n"
        f"Already-known sort: {json.dumps(existing_sort)}, limit: {json.dumps(existing_limit)}\n\n"
        "Use null for anything not mentioned — never guess, invent, or repeat an "
        "already-filled value unless the user is clearly correcting it. Normalize "
        "any date to YYYY-MM-DD; if a date has no year, assume the current year; if "
        'ambiguous and purely numeric (e.g. "7-9-2026"), interpret it as DD-MM-YYYY.\n'
        "Respond ONLY with a JSON object with exactly these keys, nothing else:\n"
        + json.dumps(schema)
    )
    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_text},
        ],
        max_tokens=300,
        response_format={"type": "json_object"},
    )

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

    if not isinstance(parsed, dict):
        return {}
    if record_type == "travel_request" and "status" in parsed:
        parsed["status"] = _normalize_travel_status(parsed.get("status"))
    if "sort" in parsed:
        parsed["sort"] = _normalize_sort(parsed.get("sort"))
    if "limit" in parsed:
        parsed["limit"] = _normalize_limit(parsed.get("limit"))
    return parsed


def process_query_request(
    text: str, pending_request: Optional[dict[str, Any]], config: dict[str, str]
) -> dict[str, Any]:
    """Main entry point called from function_app.py.

    - If `pending_request["missing_field"] == "record_type"`, `text` is the
      answer to "travel request or helpdesk ticket" — classified via a
      small scoped AI call so a typo'd answer (e.g. "helpdisk") still
      resolves instead of silently re-asking the same question.
    - If `pending_request["missing_field"] == "filters"`, `text` is a
      free-text answer that may mention any combination of filters — runs
      a scoped AI extraction rather than guessing which single field it
      maps to.
    - Otherwise this is a fresh message: run full extraction.

    Returns either:
      {"status": "complete",
       "query": {"record_type": ..., "filters": {...}, "sort": ..., "limit": ...}}
    or:
      {"status": "incomplete", "extracted": {...}, "missing_field": "...",
       "follow_up_question": "..."}
    or, if the user asked to cancel a pending lookup:
      {"status": "cancelled", "message": "..."}
    """
    if pending_request and pending_request.get("missing_field") and _is_cancel_intent(text, config):
        return {
            "status": "cancelled",
            "message": "Okay, cancelled. Let me know if you'd like to search for something else.",
        }

    original_text = text
    if pending_request and pending_request.get("missing_field") == "record_type":
        extracted = pending_request.get("extracted") or {"record_type": None, "filters": {}, "sort": None, "limit": None}
        original_text = pending_request.get("original_text", text)
        prior_sort = extracted.get("sort")
        prior_limit = extracted.get("limit")
        record_type = _classify_record_type_answer(text, config)
        if record_type:
            # The original message may have already stated a filter (a date,
            # a keyword, a sort/limit hint) that got dropped when
            # record_type came back unresolved — extraction returns {}
            # filters in that case, since the filter schema itself depends
            # on which type it is. Now that the type is known, re-examine
            # BOTH the original message and this disambiguation reply
            # against that type's schema — the reply itself may also state
            # a filter (e.g. "travel request, it's about Cuba"), not just
            # answer which type it is — so nothing said in either message
            # has to be repeated.
            filters = _empty_filters(record_type)
            combined_text = f"{original_text}. {text}" if text.strip() != original_text.strip() else original_text
            sort, limit = prior_sort, prior_limit
            try:
                updates = _extract_filters_from_answer(
                    record_type, filters, combined_text, config, prior_sort, prior_limit
                )
                for key in FILTER_KEYS[record_type]:
                    value = updates.get(key)
                    if value:
                        filters[key] = value
                sort = updates.get("sort") or prior_sort
                limit = updates.get("limit") or prior_limit
            except QueryExtractionError as exc:
                logger.warning(
                    "Could not recover filters from the original message after "
                    "resolving record_type; continuing with none recovered: %s", exc,
                )
            extracted = {"record_type": record_type, "filters": filters, "sort": sort, "limit": limit}
        # else: still unresolved — validate() below will re-ask, with
        # original_text carried forward in the response below.

    elif pending_request and pending_request.get("missing_field") == "filters":
        extracted = pending_request.get("extracted") or {"record_type": None, "filters": {}, "sort": None, "limit": None}
        record_type = extracted.get("record_type")
        filters = extracted.get("filters") or _empty_filters(record_type)
        updates = _extract_filters_from_answer(
            record_type, filters, text, config, extracted.get("sort"), extracted.get("limit")
        )
        for key in FILTER_KEYS.get(record_type, ()):
            value = updates.get(key)
            if value:
                filters[key] = value
        extracted["filters"] = filters
        extracted["sort"] = updates.get("sort") or extracted.get("sort")
        extracted["limit"] = updates.get("limit") or extracted.get("limit")

    else:
        extracted = _extract_query(text, config)

    record_type = extracted.get("record_type")
    filters = extracted.get("filters") or {}
    no_filters_yet = record_type in RECORD_TYPES and not any(
        filters.get(key) for key in FILTER_KEYS.get(record_type, ())
    )
    if no_filters_yet and not (extracted.get("sort") or extracted.get("limit")) and _wants_all(text):
        # The user explicitly asked for everything rather than narrowing
        # down — a deliberate answer, not a non-answer. Complete with no
        # filters (meaning: return every record of this type) instead of
        # re-asking the same question forever. Only applies once nothing
        # else was already extracted — a real filter (or a sort/limit)
        # always takes priority.
        return {
            "status": "complete",
            "query": {"record_type": record_type, "filters": filters, "sort": None, "limit": None},
        }

    is_complete, missing_field, follow_up_question = _validate(extracted)

    if is_complete:
        return {
            "status": "complete",
            "query": {
                "record_type": extracted["record_type"],
                "filters": extracted["filters"],
                "sort": extracted.get("sort"),
                "limit": extracted.get("limit"),
            },
        }

    result = {
        "status": "incomplete",
        "extracted": extracted,
        "missing_field": missing_field,
        "follow_up_question": follow_up_question,
    }
    if missing_field == "record_type":
        # Carried forward so a later turn can recover any filter the
        # original message already stated, once record_type is known.
        result["original_text"] = original_text
    return result
