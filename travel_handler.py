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
     asked. Follow-up date answers are parsed deterministically (cheap, no
     AI call) when they're a clean single value; if that fails — or the
     answer clearly states more than the one field asked about (e.g. both
     dates, or the reason, in the same reply) — a small scoped AI call
     extracts whatever fields the answer actually mentions instead of
     silently dropping the extra information or corrupting the field with
     unparsed raw text.
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
- If a date is ambiguous and purely numeric (e.g. "7-9-2026" or "7/9/2026"), \
interpret it as DD-MM-YYYY (day before month) — never MM-DD-YYYY.
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
        if any(ch.isdigit() for ch in country):
            # Extraction occasionally folds dates/other fields into this one
            # (e.g. "Cuba from 31-aug-2026 to 7-sep-2026") — no real country
            # name contains a digit, so this always catches that case and
            # re-asks instead of silently propagating the corrupted value.
            return False, f"{i}.destinationCountry", (
                f"\"{country}\" doesn't look like just a destination country — "
                f"could you confirm just the country you're traveling to?"
            )

        start_raw = leg.get("startDate")
        if not start_raw:
            return False, f"{i}.startDate", f"What's the start date for the trip to {country}?"
        try:
            start_dt = date.fromisoformat(start_raw)
        except (ValueError, TypeError):
            return False, f"{i}.startDate", (
                f"I couldn't understand the start date for the trip to {country} — "
                f"could you confirm it?"
            )
        if start_dt < today:
            return False, f"{i}.startDate", (
                f"The start date for the trip to {country} is in the past — "
                f"could you confirm the correct start date?"
            )

        end_raw = leg.get("endDate")
        if not end_raw:
            return False, f"{i}.endDate", f"What's the end date for the trip to {country}?"
        try:
            end_dt = date.fromisoformat(end_raw)
        except (ValueError, TypeError):
            return False, f"{i}.endDate", (
                f"I couldn't understand the end date for the trip to {country} — "
                f"could you confirm it?"
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
# Step 4: stateless follow-up — apply one answer directly when it's a clean
# single value; fall back to a scoped AI extraction when it isn't.
# --------------------------------------------------------------------------

_LEFTOVER_NOISE_CHARS = " ,.;:!?-"

_FOLLOWUP_FIELDS = ("destinationCountry", "startDate", "endDate", "reason")


def _try_parse_clean_date(answer_text: str) -> Optional[str]:
    """Returns an ISO date string only if the answer is (essentially) just
    a date with nothing meaningful left over. Returns None — rather than
    guessing — if parsing fails outright, or leftover text suggests the
    answer also states something else (another date, the reason, etc.).
    """
    try:
        # dayfirst=True: an all-numeric date like "7-9-2026" is ambiguous,
        # and dateutil otherwise defaults to US-style MM-DD-YYYY, silently
        # swapping day and month from what the user meant.
        parsed, leftover_tokens = date_parser.parse(answer_text, dayfirst=True, fuzzy_with_tokens=True)
    except (ValueError, OverflowError, TypeError):
        return None
    leftover = "".join(leftover_tokens).strip(_LEFTOVER_NOISE_CHARS).strip()
    if leftover:
        return None
    return parsed.date().isoformat()


def _extract_followup_fields(
    leg: dict[str, Any], field: str, answer_text: str, config: dict[str, str]
) -> dict[str, Any]:
    """Scoped fallback used only when the deterministic date parse can't
    cleanly consume the answer on its own (e.g. it also states the reason,
    or both dates at once). Extracts whatever fields the answer actually
    states — never guesses at ones it doesn't mention.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "The user was asked a single question about one field of a partially "
        "collected travel request, but their answer may state more than just "
        "that field.\n\n"
        f"Current known trip details (null means still missing): {json.dumps(leg)}\n"
        f'The question asked was about: "{field}"\n\n'
        "Extract any of the following fields the answer actually states. Use "
        "null for anything not mentioned — never guess, invent, or repeat an "
        "already-filled value unless the user is clearly correcting it. "
        "Normalize any date to YYYY-MM-DD. If a date is ambiguous and purely "
        'numeric (e.g. "7-9-2026" or "7/9/2026"), interpret it as DD-MM-YYYY '
        "(day before month) — never MM-DD-YYYY.\n"
        "Respond ONLY with a JSON object matching this schema, nothing else:\n"
        "{\n"
        '  "destinationCountry": string or null,\n'
        '  "startDate": string or null,\n'
        '  "endDate": string or null,\n'
        '  "reason": string or null\n'
        "}"
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
            raise TravelExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raw_output = _with_retries(_call)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise TravelExtractionError(f"Follow-up extraction did not return valid JSON: {exc}") from exc

    return parsed if isinstance(parsed, dict) else {}


def _apply_answer_via_ai(leg: dict[str, Any], field: str, answer_text: str, config: dict[str, str]) -> None:
    try:
        updates = _extract_followup_fields(leg, field, answer_text, config)
    except TravelExtractionError as exc:
        # AI fallback failed too (e.g. the model is down). Fall back to the
        # original safe behavior: leave the raw text in the asked field so
        # validation catches the bad format and re-prompts, rather than
        # losing the answer entirely.
        logger.warning("Follow-up AI extraction failed, falling back to raw text: %s", exc)
        leg[field] = answer_text
        return

    for key in _FOLLOWUP_FIELDS:
        value = updates.get(key)
        if value:
            leg[key] = value

    if not leg.get(field):
        # The AI fallback didn't manage to fill the field that was actually
        # asked about — keep the raw text there so validation re-prompts
        # with a clear "couldn't understand" message instead of looping
        # silently with no new information.
        leg[field] = answer_text


def _apply_answer(
    extracted: list[dict[str, Any]], missing_field_path: str, answer_text: str, config: dict[str, str]
) -> list[dict[str, Any]]:
    answer_text = answer_text.strip()
    idx_str, field = missing_field_path.split(".")
    idx = int(idx_str)

    while len(extracted) <= idx:
        extracted.append({"destinationCountry": None, "startDate": None, "endDate": None, "reason": None})

    if field in ("startDate", "endDate"):
        parsed = _try_parse_clean_date(answer_text)
        if parsed is not None:
            extracted[idx][field] = parsed
            return extracted
    elif not any(ch.isdigit() for ch in answer_text):
        # destinationCountry / reason: a plain-text answer with no digits is
        # trusted as-is — neither field ever legitimately needs a number.
        extracted[idx][field] = answer_text
        return extracted

    # The answer wasn't a clean fit for the field that was asked about — an
    # unparseable/absent date, or a country/reason answer containing a
    # number (almost always a sign it also states a date, e.g. "Cuba from
    # 31-aug-2026 to 7-sep-2026" when only asked for the destination). Fall
    # back to a scoped AI extraction instead of guessing or storing raw text
    # that would just get rejected — and re-asked for verbatim — later.
    _apply_answer_via_ai(extracted[idx], field, answer_text, config)
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
        extracted = _apply_answer(extracted, pending_request["missing_field_path"], text, config)
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
