"""
Travel request handler.
------------------------

Implements the travel_v1 schema as single-shot extraction rather than a
field-by-field form walk. No approver field — a request is "extracted" as a
flat list of legs:

  [ { "destinationCountry": "...", "startDate": "YYYY-MM-DD" or
      "YYYY-MM-DDTHH:mm", "endDate": same, "reason": "...",
      "requestFor": "..." or null }, ... ]

requestFor is null for the ordinary case — the requester is traveling
themselves — and is never asked about unless the conversation itself
signals otherwise. See design decision 12 below.

startDate/endDate are plain dates — "YYYY-MM-DD" — and that's all that's
ever asked for; time is not collected in this conversation (see design
decision 11). If a message happens to volunteer a time anyway (e.g. "5
Aug 2027 9:30am"), it's still captured as "YYYY-MM-DDTHH:mm" rather than
discarded, but a date with no time is just as complete — never blocked
or re-asked over a missing time.

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
     asked. Every follow-up answer — including startDate/endDate — goes
     through a small scoped AI call rather than a deterministic parser;
     a library date/time parser would confidently misread a bare
     time-only reply ("9:30am" answering "what time does it start?") as
     a fresh date defaulting to today, which is exactly the kind of
     false-confidence mistake the ambiguous-time rule below exists to
     prevent. The AI call is given the leg's current known value for the
     field being asked about, so it can merge a bare time onto an
     already-known date instead of treating it as unrelated.
  5. Date sanity checks happen in code: endDate (its date part) not
     before startDate's, startDate not before today.
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
  11a. Country approval is checked in two steps: an exact (case-
     insensitive) match against the approved list first; if that fails,
     an AI call decides whether the name is a rewritten/reformatted
     version of one of the list's entries rather than a genuinely
     different destination — e.g. extraction's own "normalize to
     standard English name" instruction rewriting the list's exact entry
     "Gaza / Palestinian Territories" into "Palestinian Territories" or
     "Palestine", which would otherwise fail a strict match even though
     the user typed back exactly what the bot itself suggested. On a
     match, the leg's destinationCountry is corrected to the list's exact
     text. Cached by normalized input (see _country_match_cache), and —
     like the business-reason check — does NOT fail open: a failed AI
     call propagates as a system error rather than silently allowing an
     unresolved country through.
  11. startDate/endDate only ever need the date — "What's the start date
     for the trip to {country}?" — time is not asked about at all
     (dropped 2026-09-11: in practice users don't give it when asked,
     so asking just adds friction for no benefit). A leg is complete
     once its dates are known, whether or not a time is attached. If a
     message happens to volunteer a time anyway (e.g. "5 Aug 2027
     9:30am"), _parse_leg_moment still recognizes and keeps the combined
     "YYYY-MM-DDTHH:mm" value rather than discarding it — this just
     means nothing is ever asked or blocked on to get one. See
     _is_valid_hhmm and _parse_leg_moment.
  12. requestFor captures a request being raised for someone other than
     the requester — "for my manager", "on behalf of my friend", "for
     John". It's extracted opportunistically wherever it's mentioned
     (the first message, or buried inside a reply that was actually
     answering something else — see _FOLLOWUP_FIELDS), never asked about
     directly on its own. When requestFor is null (the overwhelming
     majority of requests), NOTHING changes from the ordinary flow — no
     extra question, no extra field to fill. Once triggered, an actual
     EMAIL address is always required to consider it resolved — a bare
     name or relation ("John", "my manager") isn't enough on its own to
     uniquely identify who to book the trip for, so it's followed by one
     more targeted question asking specifically for the email
     ("Could you share John's email address?") rather than the broad
     one repeating itself. An email given anywhere, including inline in
     the original message ("for john@company.com"), is accepted
     immediately with no question at all — see _is_vague_on_behalf_reference
     for the broad case and the deterministic "@" shortcut in
     _apply_answer for resolving the email once asked (an email reply is
     never routed through the AI extraction path — a bare email with no
     surrounding words is exactly the kind of reply that's easy for a
     model to fail to parse, so this is handled in code instead).
     Checked FIRST — before reason, before destination — establishing
     who the trip is actually for takes priority over why or where,
     whenever it comes up at all.
"""

import json
import logging
import os
import time
from datetime import date
from typing import Any, Optional

import requests

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
      "reason": string or null,
      "requestFor": string or null
    }}
  ]
}}

Rules:
- Today's date is {today}. If a date does not mention a year, assume the \
current year from today's date above — never guess a different year.
- startDate/endDate each carry BOTH the date and the time of that leg \
boundary in one value — "YYYY-MM-DDTHH:mm" (24-hour, e.g. "2027-08-05T09:30") \
when both are known. If only the date was given, return just "YYYY-MM-DD" \
— that's a legitimate value, NOT an error; never invent a time to fill it \
in. If only a time is stated for a boundary whose date is given elsewhere \
in the same message, attach it to that date rather than leaving it \
unused — but if a time is stated with NO date anywhere for that boundary, \
leave the field null entirely; never invent or default to today's date \
(or any other date) just because a time was mentioned with nothing to \
attach it to. Normalize spoken times to 24-hour no matter how they were phrased \
— "9:30am" -> "09:30", "half nine" (evening) -> "21:30", "6 in the \
evening" / "6pm" -> "18:00", "noon" -> "12:00". If a time is genuinely \
ambiguous — "at 7" with no AM/PM, "sometime in the morning" — omit the \
time and return date-only (or null if there's no date either) rather than \
guessing; a wrong guess is worse than asking again.
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
- requestFor: if the message indicates this trip is being arranged for \
someone OTHER than the person sending the message — "for my manager", \
"on behalf of my friend", "for John", "for john@company.com", "my \
colleague needs to travel", "can I raise this for a team member" — set \
requestFor to identify who it's for: an email address EXACTLY as written \
if one was given (e.g. "john@company.com" -> "john@company.com" — never \
shorten it to a guessed name like "John"), otherwise their name if one \
was given (e.g. "John"), otherwise the relation/word actually used (e.g. \
"my manager", "my colleague", "someone"). If the message says nothing \
suggesting the trip is for anyone other than the sender, leave \
requestFor null — this is the default, ordinary case.
- The words "travel"/"trip" used only to describe the ACT of traveling \
(e.g. "I want to travel", "I need to take a trip", "raise a travel \
request") do NOT by themselves state a reason — reason must name an \
actual PURPOSE (a meeting, a conference, "business", a client visit, \
etc.), not just repeat that the message is about travel. "I want to \
travel" -> reason: null (no purpose given, only intent). "I want to \
travel for business" -> reason: "business" (a purpose was given).
- Only return an empty "legs" array if the message states NO trip detail at \
all — no destination, no dates/times, no reason, and no indication it's \
for someone else. If any one of those is stated, create one leg entry \
with that detail filled in and the rest null, even with no destination \
named."""


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
            "requestFor": leg.get("requestFor"),
        })

    return normalized_legs


# --------------------------------------------------------------------------
# Step 2 & 5: validation in code, including date sanity checks
# --------------------------------------------------------------------------

def _is_valid_hhmm(value: str) -> bool:
    """True only for a strict 24-hour "HH:mm" string (e.g. "09:30",
    "18:00")."""
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        return False
    hours, _, minutes = value.partition(":")
    if not (hours.isdigit() and minutes.isdigit()):
        return False
    return 0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59


_VAGUE_ON_BEHALF_TERMS = {
    "someone", "some one", "somebody", "some body", "another person", "other person",
    "another employee", "colleague", "colleagues", "friend", "friends", "coworker",
    "coworkers", "co-worker", "co-workers", "team member", "team members",
    "manager", "managers", "boss", "employee", "employees", "staff member",
    "staff members", "him", "her", "them", "someone else", "some one else",
}


def _is_vague_on_behalf_reference(value: str) -> bool:
    """True if `value` (a leg's requestFor) names no actual person — just a
    placeholder like "someone", "my manager", "one of my employees" — with
    no name or email attached. Used to decide whether the one on-behalf-of
    follow-up question is actually needed: a real name ("John") or an
    email is accepted immediately with no question at all, but a bare
    relation word isn't enough to identify who the trip is actually for.
    """
    if "@" in value:
        return False
    normalized = value.strip().lower()
    for prefix in ("one of my ", "one of the ", "my ", "a ", "an ", "the "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized in _VAGUE_ON_BEHALF_TERMS


def _parse_leg_moment(value: Any) -> Optional[tuple[date, Optional[str]]]:
    """Parses a combined startDate/endDate value into (date, time-or-None).
    Accepts exactly "YYYY-MM-DD" (date-only — time not yet known, a
    legitimate intermediate state) or "YYYY-MM-DDTHH:mm" (date and time
    both known). Returns None for anything else — missing, malformed, or
    the raw text _apply_answer_via_ai stores verbatim when an answer was
    unusable — so callers can tell "date-only, ask for the time" apart
    from "not a valid value at all, re-ask from scratch".
    """
    if not isinstance(value, str):
        return None
    if "T" in value:
        date_part, _, time_part = value.partition("T")
        if not _is_valid_hhmm(time_part):
            return None
        try:
            return date.fromisoformat(date_part), time_part
        except ValueError:
            return None
    try:
        return date.fromisoformat(value), None
    except ValueError:
        return None


def _leg_country_and_dates_valid(leg: dict[str, Any], today: date) -> bool:
    """True if destinationCountry/startDate/endDate are all present and
    valid for this leg (a time, if any, is not required) — reason is
    checked separately. Mirrors the same checks inline in _validate()'s
    per-leg loop below; kept as a separate helper only for the multi-leg
    reason-batching pre-check, which needs a yes/no answer rather than an
    early-return message.
    """
    country = leg.get("destinationCountry")
    if not country or any(ch.isdigit() for ch in country):
        return False
    if not country_list.is_country_approved(country):
        return False
    parsed_start = _parse_leg_moment(leg.get("startDate"))
    if parsed_start is None:
        return False
    start_dt, _ = parsed_start
    if start_dt < today:
        return False
    parsed_end = _parse_leg_moment(leg.get("endDate"))
    if parsed_end is None:
        return False
    end_dt, _ = parsed_end
    return end_dt >= start_dt


_country_match_cache: dict[str, Optional[str]] = {}


def _resolve_approved_country(country: str, config: dict[str, str]) -> Optional[str]:
    """Returns the exact approved-list entry `country` refers to, or None
    if it doesn't match any entry. Tries a cheap exact (case-insensitive)
    match first; if that fails, asks the AI whether the name is a
    rewritten/reformatted version of one of the list's entries — e.g.
    extraction's own normalization turning the list's exact entry "Gaza /
    Palestinian Territories" into "Palestinian Territories" or
    "Palestine" — rather than assuming a genuinely different destination
    was meant. Cached by normalized input text, since the same near-miss
    name recurs across a conversation and the approved list rarely
    changes within its own cache TTL. Deliberately does NOT fail open
    (unlike TopicSwitch) — same policy-gate reasoning as
    _is_business_reason: a TravelExtractionError here propagates like any
    other extraction failure rather than silently letting an unresolved
    country through.
    """
    key = country.strip().lower()
    if key in _country_match_cache:
        return _country_match_cache[key]

    approved = country_list.get_approved_countries()
    for entry in approved:
        if key == entry.strip().lower():
            _country_match_cache[key] = entry
            return entry

    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "A travel request names a destination country. Decide whether it "
        "refers to one of the EXACT entries in the approved list below, "
        "even if phrased, abbreviated, or reformatted differently — e.g. "
        'a shortened or renamed version of a multi-part entry ("Gaza / '
        'Palestinian Territories" said as just "Palestinian Territories" '
        'or "Palestine" still refers to that same list entry). If it '
        "clearly refers to one entry, respond with that entry's EXACT "
        "text, copied verbatim from the list below — never a paraphrase "
        "or a name not in the list. If it does not match any entry (a "
        "genuinely different country), respond with null.\n\n"
        f"Approved list:\n{json.dumps(approved)}\n\n"
        'Respond ONLY with a JSON object: {"match": "<exact entry text '
        'from the list>" or null}'
    )
    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": country},
        ],
        max_tokens=200,
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
        raise TravelExtractionError(f"Country-match check did not return valid JSON: {exc}") from exc

    match = parsed.get("match")
    resolved = match if isinstance(match, str) and match in approved else None
    _country_match_cache[key] = resolved
    return resolved


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
    one thing at a time, in this fixed order: requestFor (only if
    triggered — see design decision 12), reason, destination, start
    date, end date (time is not asked about — see design decision 11).
    Reason is asked when destination is still
    missing — same as every other not-yet-known field. The one
    exception: a destination that was ALREADY stated is validated
    (digit-guard, then the approved-country list) before reason or
    requestFor is asked about at all, so an invalid destination given up
    front is rejected
    on the very first message rather than only after reason is answered.
    A personal reason (e.g. a marriage proposal) is rejected and re-asked
    (checked via _is_business_reason) right after the "is it present"
    check.

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
            # A destination that was already stated is checked for validity
            # BEFORE reason is ever asked about — no point asking for the
            # reason on a trip to a destination that's going to be
            # rejected anyway. This only applies when the destination is
            # already known; if it isn't, reason is still asked first (see
            # below), same as every other still-missing field.
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
            # CountryListError (SharePoint/Graph unreachable or
            # misconfigured) intentionally propagates out of here rather
            # than being caught - a failed check must never be treated
            # as "allow it".
            resolved_country = _resolve_approved_country(country, config)
            if resolved_country is None:
                return False, f"{i}.destinationCountry", (
                    f"Sorry, {country} isn't a country we currently support for "
                    f"travel requests — could you confirm a different destination?"
                )
            if resolved_country != country:
                # Extraction rewrote the list's exact entry into something
                # that no longer matches it verbatim (e.g. "Gaza /
                # Palestinian Territories" -> "Palestinian Territories")
                # but _resolve_approved_country confirmed it's the same
                # entry — correct it to the list's exact text so it's
                # stored right and every later reference to {country}
                # (dates, reason) uses the real name.
                leg["destinationCountry"] = resolved_country
                country = resolved_country

        request_for = leg.get("requestFor")
        if request_for and "@" not in request_for:
            # An email is always required once the conversation signals
            # this trip is for someone else — a name or relation alone
            # ("John", "my manager") isn't enough to uniquely identify
            # who to book it for. Checked before reason: who the trip is
            # for is established before why, when this comes up at all.
            if _is_vague_on_behalf_reference(request_for):
                # No name or relation was even given — ask broadly.
                return False, f"{i}.requestFor", (
                    "Who is this travel request for? Could you share their "
                    "name or email address?"
                )
            # A name (or similar) is already known — ask specifically for
            # the email rather than repeat the same broad question.
            return False, f"{i}.requestFor", (
                f"Could you share {request_for}'s email address?"
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
        parsed_start = _parse_leg_moment(start_raw)
        if parsed_start is None:
            return False, f"{i}.startDate", (
                f"I couldn't understand the start date for the trip to {country} — "
                f"could you confirm it?"
            )
        start_dt, _start_time = parsed_start
        if start_dt < today:
            return False, f"{i}.startDate", (
                f"The start date for the trip to {country} is in the past — "
                f"could you confirm the correct start date?"
            )

        end_raw = leg.get("endDate")
        if not end_raw:
            return False, f"{i}.endDate", f"What's the end date for the trip to {country}?"
        parsed_end = _parse_leg_moment(end_raw)
        if parsed_end is None:
            return False, f"{i}.endDate", (
                f"I couldn't understand the end date for the trip to {country} — "
                f"could you confirm it?"
            )
        end_dt, _end_time = parsed_end
        if end_dt < start_dt:
            return False, f"{i}.endDate", (
                f"The end date for the trip to {country} is before the start date — "
                f"could you confirm the correct end date?"
            )

    return True, None, None


# --------------------------------------------------------------------------
# Step 4: stateless follow-up — every answer goes through a small scoped
# AI extraction, given the leg's current known values for context.
# --------------------------------------------------------------------------

_FOLLOWUP_FIELDS = ("destinationCountry", "startDate", "endDate", "reason", "requestFor")


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
        "counts as abandoning the question. This also applies to a bare "
        "personal-activity phrase with no connecting words — e.g. "
        '"watch movie" answering "what\'s the reason for the trip?" IS an '
        "attempt to answer (a personal, not business, reason — but still "
        'an answer: is_answer true), the same as "to watch a movie" or '
        '"watching a movie" would be; missing "to"/"a" doesn\'t make it a '
        "topic switch.\n\n"
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
    """Scoped extraction for a single follow-up answer. Extracts whatever
    fields the answer actually states — never guesses at ones it doesn't
    mention. Given the leg's current known values so it can merge a
    bare time onto a date already held for the field being asked about,
    rather than treating the two as unrelated.
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
        f'The message you are given below IS the user\'s direct reply to that '
        f'question — it is new information, not a repeat of what\'s already '
        f'known. If it states a value for "{field}", that value REPLACES '
        f"whatever \"current known trip details\" above already shows for "
        f'"{field}" (which may be empty, or may hold an earlier vague or '
        f"incomplete value that this reply is specifically meant to "
        f"resolve) — extract the NEW value from the reply itself, never "
        f"echo back the old one.\n\n"
        "Extract any of the following fields the answer actually states. Use "
        "null for anything not mentioned — never guess, invent, or repeat an "
        "already-filled value unless the user is clearly correcting it. "
        "Normalize destinationCountry to its standard English name and "
        'spelling, correcting obvious typos (e.g. "phillipines" -> '
        '"Philippines") — correct the spelling of the country that was '
        "mentioned, never substitute a different country.\n\n"
        "startDate/endDate each carry BOTH the date and the time in one "
        'value — "YYYY-MM-DDTHH:mm" (24-hour, e.g. "2027-08-05T09:30") once '
        "both are known for that boundary. If a date does not mention a "
        "year, assume the current year from today's date above — never "
        "guess a different year. If a date is ambiguous and purely numeric "
        '(e.g. "7-9-2026" or "7/9/2026"), interpret it as DD-MM-YYYY (day '
        "before month) — never MM-DD-YYYY. Normalize spoken times to "
        '24-hour no matter how phrased — "9:30am" -> "09:30", "half nine" '
        '(evening) -> "21:30", "6 in the evening" / "6pm" -> "18:00", '
        '"noon" -> "12:00".\n\n'
        f'The field being asked about ("{field}") may already hold a '
        "date-only value (see current known trip details above) — if the "
        "answer gives ONLY a time and no date of its own, AND that field "
        "already has a date, combine the new time with the date ALREADY "
        'KNOWN for that same field to produce the full "YYYY-MM-DDTHH:mm" '
        "value, rather than leaving it out or treating it as unrelated. "
        "If the answer gives ONLY a time and the field does NOT already "
        "have a date (current known trip details shows null for it), "
        "return null for that field instead — never invent or default to "
        "today's date (or any other date) just because a time was given "
        "with nothing to attach it to; a date must actually be stated or "
        "already known before a time means anything. If the answer gives a date with no time "
        'yet, return just "YYYY-MM-DD" — that\'s a legitimate value, not '
        "an error, never invent a time to complete it. If a time is "
        'genuinely ambiguous — "at 7" with no AM/PM given, "sometime in '
        'the morning" — return null for that field so the date-only value '
        "already held is left untouched and gets asked about again; a "
        "wrong guess is worse than asking again.\n\n"
        "requestFor: if the answer indicates this trip is being arranged "
        'for someone OTHER than the person replying — "for my manager", '
        '"on behalf of my friend", "for John", "for john@company.com", '
        '"not me, for my friend", "my colleague needs to travel" — set '
        "requestFor to identify who it's for: an email address EXACTLY "
        'as written if one was given (e.g. "john@company.com" -> '
        '"john@company.com" — never shorten it to a guessed name like '
        '"John"), otherwise their name if given (e.g. "John"), otherwise '
        'the relation/word used (e.g. "my manager", "someone"). '
        "This applies even when the question being asked was about a "
        "completely different field (e.g. answering a destination "
        'question with "japan, not me, for my friend" states BOTH the '
        "destination AND requestFor — extract both). Leave requestFor "
        "null if the answer says nothing suggesting the trip is for "
        "anyone else.\n\n"
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
        '  "requestFor": string or null,\n'
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


def _apply_answer_via_ai(
    leg: dict[str, Any], field: str, answer_text: str, config: dict[str, str],
    already_confirmed_answer: bool = False,
) -> None:
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

    if (
        updates.get("is_topic_switch")
        and not any(updates.get(key) for key in _FOLLOWUP_FIELDS)
        and not already_confirmed_answer
    ):
        # Flagged as unrelated AND nothing usable was extracted from it —
        # a real (if messy) answer would have yielded at least one field.
        # Skipped when the caller already ran the dedicated TopicSwitch
        # check and it confirmed this IS a real answer attempt (e.g. a
        # personal-but-stated reason like "watch movie") — that check's
        # prompt is tuned specifically for this judgment call, and this
        # extraction call's own is_topic_switch flag is secondary to it,
        # not a second vote that can override it.
        raise TopicSwitch()

    for key in _FOLLOWUP_FIELDS:
        value = updates.get(key)
        if value:
            leg[key] = value

    if not updates.get(field):
        if field in ("startDate", "endDate", "requestFor"):
            # Deliberately leave the existing value untouched rather than
            # overwrite it with raw, potentially unrelated text:
            # - startDate/endDate: the ambiguous-time case (see
            #   _extract_followup_fields) — extraction returns null on
            #   purpose rather than invent a time, and the date-only value
            #   already held must survive so the same clean re-ask goes
            #   out again next turn.
            # - requestFor: a reply that doesn't clearly identify a person
            #   (e.g. it actually answers a DIFFERENT field, like stating
            #   the reason instead) must not get stored as if it WERE a
            #   resolved identity — that would silently corrupt the data
            #   with garbage that happens to not match the vague-reference
            #   stoplist. Keeping the existing (still-vague) value and
            #   re-asking is safe; accepting anything at all is not.
            pass
        else:
            # The AI didn't manage to extract this field from the answer.
            # Store the raw text so the NEXT validation pass re-evaluates
            # something fresh, rather than silently keeping whatever was
            # already there — which matters whenever the field being asked
            # about already held a value that's specifically why it's
            # being re-asked (e.g. a reason/destination re-asked for
            # having been rejected): without this, a failed extraction on
            # the new answer would leave the old value in place forever,
            # re-asking the same question on every subsequent turn with no
            # way to progress.
            leg[field] = answer_text


def _apply_answer(
    extracted: list[dict[str, Any]], missing_field_path: str, follow_up_question: str,
    answer_text: str, config: dict[str, str],
) -> list[dict[str, Any]]:
    answer_text = answer_text.strip()
    idx_str, field = missing_field_path.split(".")

    if idx_str == "*":
        # Shared answer for every leg still missing this field (currently
        # only "*.reason" — see _validate). A reason never legitimately
        # contains a digit; if this one does, something's off with the
        # answer, so fall back to the normal single-leg path (against the
        # first leg still missing it) rather than copying something
        # suspicious onto every leg at once.
        idx = next((i for i, leg in enumerate(extracted) if not leg.get(field)), 0)
    else:
        idx = int(idx_str)

    while len(extracted) <= idx:
        extracted.append({
            "destinationCountry": None, "startDate": None, "endDate": None,
            "reason": None, "requestFor": None,
        })

    if field == "requestFor" and "@" in answer_text:
        # Deterministic shortcut, checked before anything else: an email
        # address answering "who is this for" / "could you share their
        # email" is completely unambiguous — no AI call needed, and
        # critically none of the risk that one fails to parse a bare
        # email reply with no surrounding words (the model has nothing to
        # reason about, so this is exactly where extraction is most
        # likely to come back empty and, before this shortcut existed,
        # get the reply silently ignored and the same question re-asked).
        # Combines with an existing concrete name if there is one, e.g.
        # "John" + "john@company.com" -> "John, john@company.com";
        # otherwise the email stands alone.
        existing = extracted[idx].get("requestFor")
        if existing and "@" not in existing and not _is_vague_on_behalf_reference(existing):
            extracted[idx]["requestFor"] = f"{existing}, {answer_text}"
        else:
            extracted[idx]["requestFor"] = answer_text
        return extracted

    # destinationCountry / reason / requestFor: a plain-text answer with no
    # digits gets the TopicSwitch check (neither field ever legitimately
    # needs a number — an email answering requestFor is handled by the
    # deterministic shortcut above and never reaches here) and is then
    # routed through the scoped AI extraction below rather than trusted
    # verbatim — TopicSwitch only confirms the reply is a real attempt to
    # answer, not that the reply IS the value itself (e.g. "i want to
    # travel to cuba" is a real answer, but "Cuba" — not the whole
    # sentence — is the actual destination).
    trusts_directly = (
        field in ("destinationCountry", "reason", "requestFor")
        and not any(ch.isdigit() for ch in answer_text)
    )
    if trusts_directly and _is_topic_switch(follow_up_question, answer_text, config):
        raise TopicSwitch()

    if trusts_directly:
        # Don't trust the reply verbatim, even though TopicSwitch already
        # confirmed it's a real attempt to answer — it may be a full
        # sentence containing the value rather than just the value itself
        # (e.g. "i want to travel to cuba" answering "confirm a different
        # destination?"), which would otherwise get stored as the literal
        # destination string instead of "Cuba" being extracted from it.
        # Extract properly instead, the same way fresh extraction would.
        # already_confirmed_answer=True: the dedicated TopicSwitch check
        # above already confirmed this is a real answer attempt, so this
        # call's own (secondary, less-tuned) is_topic_switch flag must
        # not override that and reject it a second time.
        _apply_answer_via_ai(extracted[idx], field, answer_text, config, already_confirmed_answer=True)
        if idx_str == "*":
            value = extracted[idx].get(field)
            if value:
                for leg in extracted:
                    if not leg.get(field):
                        leg[field] = value
        return extracted

    # Everything else — startDate/endDate always, plus a country/reason
    # answer containing a number (almost always a sign it also states a
    # date, e.g. "Cuba from 31-aug-2026 to 7-sep-2026" when only asked
    # for the destination) — goes through the same scoped AI extraction.
    # No deterministic fast path for dates: a bare time-only reply
    # ("9:30am" answering a time-only re-ask) would otherwise be
    # misparsed as today's date by a library parser instead of being
    # merged onto the date already held for this field.
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
