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
  6. If a message names multiple destinations and every leg is otherwise
     complete, a single missing reason is asked about once for all of
     them ("What's the reason for the trips?") rather than once per leg —
     the answer applies to every leg still missing it.
  7. destinationCountry is checked against an approved-country list
     fetched from SharePoint (see country_list.py) — a well-formed
     country name that isn't on that list is rejected and re-asked,
     same as the digit-guard above it. This check runs before anything
     else is asked about a leg (even before the reason) — no point
     asking for a reason or dates on a trip to a destination that will
     be rejected anyway.
  8. A reply to a pending follow-up question is checked for whether it
     actually attempts to answer that question before being trusted —
     see TopicSwitch. Free-text fields (destinationCountry, reason) had
     no such check previously and would silently absorb any unrelated
     reply (e.g. a helpdesk complaint typed while a travel request was
     mid-flow) as if it were the answer.
  9. reason is checked for being a legitimate BUSINESS reason (client
     visit, conference, training, etc.) — this system only handles
     business travel, not personal trips (a marriage proposal, a
     vacation, visiting family for personal reasons). A personal reason
     is rejected and re-asked, same pattern as the country-approval
     check. Cached by normalized reason text (see
     _business_reason_cache) since the judgment for a given reason never
     changes and _validate() re-checks it on every subsequent turn.
  10. A reply to a pending follow-up question is checked for cancel
     intent (see _is_cancel_intent) BEFORE anything else — "cancel",
     "exit", "start over", "I don't want this", and similar all end the
     in-progress request outright, rather than being mistaken for an
     off-topic reply (TopicSwitch) or, worse, literally stored as the
     answer to whichever field was being asked about.
"""

import json
import logging
import os
import time
from datetime import date
from typing import Any, Optional

import requests
from dateutil import parser as date_parser

import ai_client
import country_list

logger = logging.getLogger("travel_handler")

AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "1.5"))


class TravelExtractionError(Exception):
    """Raised when the extraction call fails or returns unusable output."""


class TopicSwitch(Exception):
    """Raised when a reply to a pending follow-up question doesn't attempt
    to answer it at all, but instead looks like the user has moved on to a
    completely different, unrelated topic (a helpdesk problem, a lookup
    request, or anything else). Caught by process_travel_request to
    surface a "you have a request in progress" notice instead of silently
    absorbing unrelated text into a travel field.
    """


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

def _build_extraction_system_prompt() -> str:
    # Built per-call (not a module-level constant) so it always reflects the
    # real current date — the model has no other way to know "today" and
    # will otherwise default an omitted year to whatever it learned during
    # training (observed: 2023) instead of the actual current year.
    today = date.today().isoformat()
    return f"""You extract structured travel request data from a user's message.

Respond ONLY with a JSON object matching this exact schema, nothing else:
{{
  "legs": [
    {{
      "destinationCountry": string or null,
      "startDate": string or null,
      "endDate": string or null,
      "reason": string or null
    }}
  ]
}}

Rules:
- Today's date is {today}. If a date does not mention a year, assume the \
current year from today's date above — never guess a different year.
- Create one entry in "legs" for each distinct trip mentioned in the message \
— a trip counts even if it only states a reason or dates so far, with no \
destination named yet; it does NOT need a destination specifically to count \
as an entry.
- Normalize every date to YYYY-MM-DD regardless of how it was written in the \
message (e.g. "26 Aug 2026" and "26-Aug-2026" both become "2026-08-26").
- If a date is ambiguous and purely numeric (e.g. "7-9-2026" or "7/9/2026"), \
interpret it as DD-MM-YYYY (day before month) — never MM-DD-YYYY.
- Normalize destinationCountry to its standard English name and spelling, \
correcting obvious typos (e.g. "phillipines" -> "Philippines", "austrailia" \
-> "Australia") — correct the spelling of the country that was mentioned, \
never substitute a different country.
- If a field is not mentioned at all, use null. Never guess or invent a value.
- The words "travel"/"trip" used only to describe the ACT of traveling \
(e.g. "I want to travel", "I need to take a trip", "raise a travel \
request") do NOT by themselves state a reason — reason must name an \
actual PURPOSE (a meeting, a conference, "business", a client visit, \
etc.), not just repeat that the message is about travel. "I want to \
travel" -> reason: null (no purpose given, only intent). "I want to \
travel for business" -> reason: "business" (a purpose was given).
- Only return an empty "legs" array if the message states NO trip detail at \
all — no destination, no dates, and no reason. If any one of those is \
stated, create one leg entry with that detail filled in and the rest null, \
even with no destination named."""


def _extract_fields(text: str, config: dict[str, str]) -> list[dict[str, Any]]:
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
        max_tokens=800,
        response_format={"type": "json_object"},
    )

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

def _leg_country_and_dates_valid(leg: dict[str, Any], today: date) -> bool:
    """True if destinationCountry/startDate/endDate are all present and
    valid for this leg — reason is checked separately. Mirrors the same
    checks inline in _validate()'s per-leg loop below; kept as a separate
    helper only for the multi-leg reason-batching pre-check, which needs a
    yes/no answer rather than an early-return message.
    """
    country = leg.get("destinationCountry")
    if not country or any(ch.isdigit() for ch in country):
        return False
    if not country_list.is_country_approved(country):
        return False
    try:
        start_dt = date.fromisoformat(leg.get("startDate"))
    except (ValueError, TypeError):
        return False
    if start_dt < today:
        return False
    try:
        end_dt = date.fromisoformat(leg.get("endDate"))
    except (ValueError, TypeError):
        return False
    return end_dt >= start_dt


_business_reason_cache: dict[str, bool] = {}


def _is_business_reason(reason: str, config: dict[str, str]) -> bool:
    """Is this a legitimate BUSINESS travel reason, or something clearly
    personal (a marriage proposal, a vacation, visiting family/friends for
    personal reasons, a honeymoon, etc.)? This system only accepts business
    travel requests. Cached by normalized reason text — the judgment for a
    given reason never changes, and _validate() re-checks every leg's
    reason on every subsequent turn, so this avoids repeating the same AI
    call for a reason that's already been confirmed. Deliberately does NOT
    fail open on an error (unlike TopicSwitch) — this is a policy gate
    like the country-approval check, not just a UX nicety, so a
    TravelExtractionError here propagates like any other extraction
    failure rather than silently letting a personal reason through.
    """
    key = reason.strip().lower()
    if key in _business_reason_cache:
        return _business_reason_cache[key]

    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "A corporate travel request system only accepts BUSINESS travel "
        "reasons — client visits, conferences, meetings, work-related "
        "training, site visits, business development, and similar. It "
        "does NOT accept personal travel reasons — a marriage proposal, "
        "a vacation, visiting family/friends for personal reasons, a "
        "honeymoon, and any form of personal leisure or entertainment "
        "(watching a movie or show, attending a concert or sporting "
        "event as a spectator, shopping, sightseeing, and similar), "
        "even when phrased formally or with travel-sounding words.\n\n"
        "A single generic word is still acceptable on its own, even with "
        "no further detail — the later destination/date questions cover "
        "specifics, this check only needs to rule out something CLEARLY "
        'personal or meaningless: "travel", "trip", "business", '
        '"meeting", "conference", "training", "work" are all acceptable '
        "as-is, not rejected for being vague. Reject only when the "
        "reason is clearly personal (a vacation, a proposal, visiting "
        "family) — never merely because it's short or generic. Tolerate "
        "typos and casual phrasing when judging this — a misspelled "
        'business reason (e.g. "cleint vist" for "client visit") is '
        "still business, not a reason to reject it.\n\n"
        "Decide whether the given trip reason is a legitimate business "
        "reason.\n\n"
        'Respond ONLY with a JSON object: {"is_business": true or false}'
    )
    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reason},
        ],
        max_tokens=20,
        response_format={"type": "json_object"},
    )

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
        raise TravelExtractionError(f"Business-reason check did not return valid JSON: {exc}") from exc

    is_business = parsed.get("is_business") is not False
    _business_reason_cache[key] = is_business
    return is_business


def _validate(extracted: list[dict[str, Any]], config: dict[str, str]) -> tuple[bool, Optional[str], Optional[str]]:
    """Returns (is_complete, missing_field_path, follow_up_question).

    Checks each leg in turn and returns the FIRST problem found — we ask
    one thing at a time. No approver field — travel requests only need
    destination, dates, and reason per leg. A destination that was already
    given is checked for validity (digit-guard, then the approved-country
    list) FIRST, before anything else — no point asking for the reason or
    dates on a trip to a destination that's going to be rejected anyway.
    Once the destination is confirmed valid (or is absent), the usual
    priority applies: reason, then destination, then dates. A given reason
    is checked for being a legitimate business reason right after the
    "is it present" check — a personal reason (e.g. a marriage proposal)
    is rejected and re-asked, same pattern as the destination checks.

    Exception: if every leg already has a valid destination/dates and 2+
    of them are missing only the reason, ask once for all of them
    ("What's the reason for the trips?", missing_field_path "*.reason")
    instead of once per leg — the same trip purpose commonly covers every
    destination named in one message, and repeating the question per leg
    would just make the user type the same answer twice.
    """
    if not extracted:
        return False, "0.reason", "What's the reason for your trip?"

    today = date.today()

    if len(extracted) >= 2 and all(_leg_country_and_dates_valid(leg, today) for leg in extracted):
        legs_missing_reason = sum(1 for leg in extracted if not leg.get("reason"))
        if legs_missing_reason >= 2:
            return False, "*.reason", "What's the reason for the trips?"

    for i, leg in enumerate(extracted):
        country = leg.get("destinationCountry")

        if country:
            if any(ch.isdigit() for ch in country):
                # Extraction occasionally folds dates/other fields into this
                # one (e.g. "Cuba from 31-aug-2026 to 7-sep-2026") — no real
                # country name contains a digit, so this always catches that
                # case and re-asks instead of silently propagating the
                # corrupted value.
                return False, f"{i}.destinationCountry", (
                    f"\"{country}\" doesn't look like just a destination country — "
                    f"could you confirm just the country you're traveling to?"
                )
            if not country_list.is_country_approved(country):
                # CountryListError (SharePoint/Graph unreachable or
                # misconfigured) intentionally propagates out of here rather
                # than being caught - a failed check must never be treated
                # as "allow it".
                return False, f"{i}.destinationCountry", (
                    f"Sorry, {country} isn't a country we currently support for "
                    f"travel requests — could you confirm a different destination?"
                )

        reason = leg.get("reason")
        if not reason:
            if country:
                return False, f"{i}.reason", f"What's the reason for the trip to {country}?"
            return False, f"{i}.reason", f"What's the reason for trip #{i + 1}?"
        if not _is_business_reason(reason, config):
            return False, f"{i}.reason", (
                "Sorry, this system only accepts business travel requests — "
                "could you provide the business reason for the trip"
                f"{f' to {country}' if country else ''}?"
            )

        if not country:
            if i == 0:
                return False, f"{i}.destinationCountry", "Where are you traveling to?"
            return False, f"{i}.destinationCountry", f"What's the destination country for trip #{i + 1}?"

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


def _is_cancel_intent(text: str, config: dict[str, str]) -> bool:
    """Does this reply express a clear intent to CANCEL/ABORT/STOP the
    in-progress travel request entirely — not just decline to answer the
    current question, but end the whole process? Checked before anything
    else once a travel request is pending, so a cancellation short-
    circuits the whole apply-answer/topic-switch pipeline instead of
    being mistaken for an off-topic reply, or worse, literally stored as
    the answer to whichever field was being asked about. Fails open
    (treats as NOT a cancellation) if the check itself errors out — a
    broken check must never block a legitimate answer.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "The user has a travel request in progress. Decide whether their "
        "reply expresses a clear intent to CANCEL/ABORT/STOP the request "
        "entirely — not merely decline to answer the current question, "
        'but actually end the whole process. Words like "cancel", '
        '"close", "exit", "quit", "abort", "discard", "discard request", '
        '"start over", "don\'t continue", "I don\'t want this", or '
        "similar all count, tolerating typos and casual phrasing. A "
        "reply that attempts to answer the question, even vaguely, or "
        "that raises an unrelated topic without asking to stop, does "
        "NOT count.\n\n"
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
            raise TravelExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (TravelExtractionError, json.JSONDecodeError) as exc:
        logger.warning("Cancel-intent check failed, treating as not a cancellation: %s", exc)
        return False

    return parsed.get("is_cancel") is True


def _is_topic_switch(follow_up_question: str, answer_text: str, config: dict[str, str]) -> bool:
    """Scoped, question-specific check used only for the fields that would
    otherwise be trusted with no validation at all (see _apply_answer).
    Deliberately NOT the general-purpose classifier in function_app.py —
    that classifier is tuned to detect NEW request intent and would
    false-positive on a plain, legitimate answer with no travel-specific
    wording (e.g. a bare reason like "family visit" reads as general_query
    under those rules, but is a perfectly good answer here). Fails open
    (treats the reply as a normal answer) if the check itself errors out —
    a broken topic-switch check must never block a legitimate answer.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "A chatbot asked the user this single question while filling out a "
        f'travel request: "{follow_up_question}"\n\n'
        "Decide whether the user's reply attempts to answer that question — "
        "even if vague, partial, or oddly worded — or whether it clearly "
        "abandons the question and raises something unrelated instead (an "
        "IT/helpdesk problem, a request to look something up, or any other "
        "unrelated topic). A short, generic word that plausibly relates to "
        'the question\'s own subject IS an attempt to answer, not a switch '
        '— e.g. replying "travel", "trip", or "training" to "what\'s the '
        'reason for the trip?" is a vague but real answer (is_answer: '
        'true), not a topic switch; only something that names a clearly '
        "different subject (a helpdesk problem, an unrelated request) "
        "counts as abandoning the question.\n\n"
        'Respond ONLY with a JSON object: {"is_answer": true or false}'
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
            raise TravelExtractionError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (TravelExtractionError, json.JSONDecodeError) as exc:
        logger.warning("Topic-switch check failed, treating the reply as a normal answer: %s", exc)
        return False

    return parsed.get("is_answer") is False


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
    today = date.today().isoformat()
    system_prompt = (
        "The user was asked a single question about one field of a partially "
        "collected travel request, but their answer may state more than just "
        "that field.\n\n"
        f"Today's date is {today}.\n"
        f"Current known trip details (null means still missing): {json.dumps(leg)}\n"
        f'The question asked was about: "{field}"\n\n'
        "Extract any of the following fields the answer actually states. Use "
        "null for anything not mentioned — never guess, invent, or repeat an "
        "already-filled value unless the user is clearly correcting it. "
        "Normalize any date to YYYY-MM-DD. If a date does not mention a year, "
        "assume the current year from today's date above — never guess a "
        "different year. If a date is ambiguous and purely numeric (e.g. "
        '"7-9-2026" or "7/9/2026"), interpret it as DD-MM-YYYY (day before '
        "month) — never MM-DD-YYYY. Normalize destinationCountry to its "
        'standard English name and spelling, correcting obvious typos (e.g. '
        '"phillipines" -> "Philippines") — correct the spelling of the '
        "country that was mentioned, never substitute a different country. "
        'Also set "is_topic_switch" to true if the reply clearly abandons '
        "the question and raises something unrelated instead (an IT/"
        "helpdesk problem, a lookup request, or any other unrelated "
        "topic) rather than attempting to answer it at all — false "
        "otherwise, including for vague, partial, or oddly worded answers.\n"
        "Respond ONLY with a JSON object matching this schema, nothing else:\n"
        "{\n"
        '  "destinationCountry": string or null,\n'
        '  "startDate": string or null,\n'
        '  "endDate": string or null,\n'
        '  "reason": string or null,\n'
        '  "is_topic_switch": true or false\n'
        "}"
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

    if updates.get("is_topic_switch") and not any(updates.get(key) for key in _FOLLOWUP_FIELDS):
        # Flagged as unrelated AND nothing usable was extracted from it —
        # a real (if messy) answer would have yielded at least one field.
        raise TopicSwitch()

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
    extracted: list[dict[str, Any]], missing_field_path: str, follow_up_question: str,
    answer_text: str, config: dict[str, str],
) -> list[dict[str, Any]]:
    answer_text = answer_text.strip()
    idx_str, field = missing_field_path.split(".")

    # destinationCountry / reason: a plain-text answer with no digits would
    # otherwise be trusted as-is with no validation at all — neither field
    # ever legitimately needs a number, so there was never an AI call in
    # this path to catch a reply that isn't actually an answer (e.g. a
    # helpdesk complaint typed mid-flow). Checked once here, ahead of both
    # the wildcard and single-leg branches below.
    trusts_directly = field in ("destinationCountry", "reason") and not any(ch.isdigit() for ch in answer_text)
    if trusts_directly and _is_topic_switch(follow_up_question, answer_text, config):
        raise TopicSwitch()

    if idx_str == "*":
        # Shared answer for every leg still missing this field (currently
        # only "*.reason" — see _validate). A reason never legitimately
        # contains a digit; if this one does, something's off with the
        # answer, so fall back to the normal single-leg path (against the
        # first leg still missing it) rather than copying something
        # suspicious onto every leg at once.
        if trusts_directly:
            for leg in extracted:
                if not leg.get(field):
                    leg[field] = answer_text
            return extracted
        idx = next((i for i, leg in enumerate(extracted) if not leg.get(field)), 0)
    else:
        idx = int(idx_str)

    while len(extracted) <= idx:
        extracted.append({"destinationCountry": None, "startDate": None, "endDate": None, "reason": None})

    if field in ("startDate", "endDate"):
        parsed = _try_parse_clean_date(answer_text)
        if parsed is not None:
            extracted[idx][field] = parsed
            return extracted
    elif trusts_directly:
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
    or, if the user asked to cancel a pending request:
      {"status": "cancelled", "message": "..."}
    """
    if pending_request and pending_request.get("missing_field_path"):
        if _is_cancel_intent(text, config):
            return {
                "status": "cancelled",
                "message": (
                    "Your travel request has been cancelled. Let me know "
                    "if you'd like to start a new one."
                ),
            }
        extracted = pending_request.get("extracted") or []
        try:
            extracted = _apply_answer(
                extracted, pending_request["missing_field_path"],
                pending_request.get("follow_up_question", ""), text, config,
            )
        except TopicSwitch:
            return {
                "status": "incomplete",
                "extracted": extracted,
                "missing_field_path": pending_request["missing_field_path"],
                "follow_up_question": (
                    "You have a travel request in progress. Please finish "
                    "answering the question below to continue it, or "
                    "refresh the chat to start a new conversation.\n\n"
                    f"{pending_request.get('follow_up_question', '')}"
                ),
            }
    else:
        extracted = _extract_fields(text, config)

    is_complete, missing_field_path, follow_up_question = _validate(extracted, config)

    if is_complete:
        return {"status": "complete", "extracted": extracted}

    return {
        "status": "incomplete",
        "extracted": extracted,
        "missing_field_path": missing_field_path,
        "follow_up_question": follow_up_question,
    }
