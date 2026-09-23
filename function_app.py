"""
Azure Function: Classify + Route
--------------------------------

HTTP-triggered function that:
  1. Validates the incoming question.
  2. Classifies it as "helpdesk" or "travel_request" using Azure OpenAI
     (gpt-4o-mini).
  3. Routes to the matching handler.

This file intentionally keeps the help-desk search logic and the
schema-driven conversation flow as clearly marked stubs (see
`handle_helpdesk_request` and `handle_travel_request`) — those are the next
two pieces to build out and plug in here. Everything around them
(validation, classification, retries, logging, response shape) is complete
and production-ready as-is.

Secrets are read from Function App settings for now (per current setup).
To move to Key Vault later, only `_get_config()` needs to change — nothing
else in this file depends on where the secrets come from.
"""

import json
import logging
import os
import time
from typing import Any, Optional

import azure.functions as func
import requests

from helpdesk_api_client import call_helpdesk_api, HelpdeskAPIConfigError, HelpdeskAPIError
import ai_client
import country_list
import query_handler
import travel_handler

# --------------------------------------------------------------------------
# Configuration (Function App settings / env vars)
# --------------------------------------------------------------------------

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")

MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "4000"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "1.5"))

# Below this confidence, we still route on the model's best guess but flag it —
# useful later for logging/analytics on ambiguous questions.
LOW_CONFIDENCE_THRESHOLD = float(os.environ.get("LOW_CONFIDENCE_THRESHOLD", "0.6"))

ALLOWED_CATEGORIES = ["helpdesk", "travel_request", "query_request", "ambiguous_request", "general_query", "cancel_request"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classify_router")

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""


class AzureOpenAIError(Exception):
    """Raised for transient Azure OpenAI failures — safe to retry (429, 5xx,
    network-level errors)."""


class AzureOpenAINonRetryableError(Exception):
    """Raised for permanent Azure OpenAI failures — retrying will never
    help (bad API key, wrong deployment name, malformed request, etc.).
    The full detail is logged server-side; the client gets a distinct,
    honest message rather than 'temporarily unavailable'.
    """


def _get_config() -> dict[str, str]:
    """Central place secrets/config are resolved from.

    Currently reads Function App settings directly. If you move to Key
    Vault later, this is the only function that needs to change — swap the
    os.environ reads for SecretClient calls and cache the result.
    """
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
        raise ConfigurationError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set in "
            "Function App settings."
        )
    return {
        "endpoint": AZURE_OPENAI_ENDPOINT,
        "api_key": AZURE_OPENAI_API_KEY,
    }


# --------------------------------------------------------------------------
# Azure OpenAI call (classification) with retry/backoff
# --------------------------------------------------------------------------

def _with_retries(fn, *args, **kwargs):
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.RequestException, AzureOpenAIError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Azure OpenAI call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
    raise AzureOpenAIError(f"Azure OpenAI call failed after {MAX_RETRIES} attempts") from last_exc


def _classify_question(text: str, config: dict[str, str]) -> dict[str, Any]:
    """Calls gpt-4o-mini to classify the question. Returns a dict with
    'category', 'confidence', and 'reasoning'.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}

    system_prompt = (
        "You classify an incoming chatbot question into exactly one of six "
        "categories. IMPORTANT: completeness is NEVER a reason to pick "
        '"ambiguous_request" or "general_query" — missing details like a '
        "destination, dates, or a specific IT symptom are always resolved "
        "later by follow-up questions, not by classification. The ONLY "
        "question you're answering here is WHICH TOPIC the message is "
        "about (travel vs. IT/helpdesk vs. neither), never whether enough "
        "detail was given.\n\n"
        "Before anything else, check: does the message express intent to "
        "CANCEL/DELETE/WITHDRAW/REMOVE/STOP a request or ticket, with no "
        'other real content — "cancel", "delete", "cancel the request", '
        '"delete request", "withdraw my request", "remove the ticket", and '
        'similar? -> "cancel_request". This matches regardless of whether '
        "a specific topic (travel/helpdesk) is also named alongside it "
        '("cancel my travel request", "delete the helpdesk ticket" are '
        'still "cancel_request") — this system has no persistence layer, '
        "so it can never look up or cancel a previously submitted record "
        "either way, meaning which type was named doesn't change the "
        "outcome. If this doesn't match, move on to the numbered rules "
        "below as normal.\n\n"
        "Decide in this exact order, stopping at the first match:\n\n"
        "1. Is the message asking for an EXPLANATION of, or INSTRUCTIONS "
        'for, the travel-request/helpdesk-ticket FEATURES THEMSELVES — '
        'phrased as a question about the concept or process (e.g. '
        '"explain the travel request concept", "how do I create a travel '
        'request", "how to get my previous request", "what\'s required to '
        'create a travel request", "what fields do I need for a helpdesk '
        'ticket") — rather than actually trying to create or look '
        'something up right now? -> "helpdesk". This takes priority over '
        'every rule below: containing "travel"/"trip", a lookup verb, or '
        "an IT term does NOT mean the message is performing that action "
        "if it's really asking HOW the feature works. Contrast with real "
        "action messages, which state or imply the person's OWN trip or "
        'lookup rather than asking about the feature in the abstract — '
        'these are NOT explanation questions and must NOT match this '
        'rule: "I want to travel to Cuba" (states personal intent to '
        'travel -> rule 3 below), "trip to meet a client" (personal '
        'intent, even with no destination given -> rule 3), "show my '
        'tickets based on Cuba" (gives real search criteria -> rule 2). '
        "However, stated intent PLUS an explicit question about HOW to "
        'do it ("how do I do that", "how to do that", "how to apply", '
        '"how to proceed") DOES match this rule despite naming a '
        "destination or the person's own trip — the trailing process "
        "question is the deciding signal, not whether a destination was "
        'mentioned: "I want to travel to Cuba, how do I do that?", "I '
        'want to travel to Cuba, how to apply?", "Ram wants to travel, '
        'how and where can he apply?" -> "helpdesk" (rule 1), NOT rule '
        '3 — contrast with the same intent stated with NO process '
        'question at all ("I want to travel to Cuba" alone -> rule 3). '
        "This also applies with NO topic named at all: a bare "
        '"how"/"where" process question about raising/creating/applying '
        'for "a request" — with no destination, no IT term, nothing — '
        'still matches this rule, NOT rule 5: "how and where can I '
        'raise a request", "how do I apply for a request", "where do I '
        'go to submit a request" -> "helpdesk" (rule 1). The presence '
        'of "how"/"where" asking about the PROCESS is what matters, not '
        "whether a topic was named — contrast with rule 5's bare create "
        'phrasing, which has NO "how"/"where" question at all (e.g. '
        '"raise a request", "request needs to be raised" — a direct '
        "attempt to create something now, not a question about how to). "
        'Separately, a "Can I"/"Could I" polite phrasing is NOT by itself '
        "a process question, even though it's grammatically a question — "
        "it's just a courteous way of making the request itself, and by "
        'default should be read EXACTLY as if "Can I"/"Could I" were '
        "deleted from the sentence, then classified by whatever rule the "
        "remaining words match. Work through rules 2-5 on what's left "
        'before ever concluding rule 1: "Can I create a request for my '
        'friend instead of myself?" / "Could I raise a request for '
        'someone else?" -> delete "Can I"/"Could I" -> "create a request '
        'for my friend..." / "raise a request for someone else" -> no '
        'topic named -> rule 5 (ambiguous_request), NOT rule 1. "Can I '
        'raise a travel request for another person?" -> delete "Can I" '
        '-> "raise a travel request for another person" -> contains '
        '"travel" -> rule 3 (travel_request), NOT rule 1 and NOT rule 5 '
        "— naming a topic always takes priority over the no-topic "
        'carve-out, "Can I" or not. The ONLY time a "Can I"/"Could I" '
        "question actually stays rule 1 is when what's left after "
        "deleting it is itself a genuine question about the MECHANICS or "
        'POLICY of an already-named feature, not a request to act: "Can I '
        'raise a travel request without manager approval?" -> delete '
        '"Can I" -> "raise a travel request without manager approval" — '
        "this is asking whether/how something works, not stating an "
        "intent to raise one right now, so it stays rule 1. The deciding "
        "question is always: once you remove the polite \"Can I\", is what "
        "remains an actual request to do the thing (-> rules 2-5, same "
        "as any other phrasing), or a genuine question about whether/how "
        "it can be done (-> rule 1)?\n"
        "2. Does the message use a LOOKUP/retrieval verb or phrase — get, "
        'show, see, check, find, list, pull up, "what\'s the status of" '
        "— about a request or ticket that's implied to ALREADY exist? "
        "The verb/phrase itself is the primary signal, regardless of "
        'which article or determiner follows it: "my request" '
        '(possessive), "the request" (definite article), "all the '
        'requests"/"all my requests"/"list all" (enumerating existing '
        'items) are ALL references to something already submitted — '
        "NEVER something being created now. This only applies when what's "
        "being looked up is itself a SUBMITTED item with a status to "
        'check — a request/ticket left unqualified, e.g. "my request", '
        '"my ticket" (always in scope by default), or a named item that '
        'clearly IS one — "travel request", "helpdesk ticket", "IT '
        'ticket", "flight ticket"/"flight status" (a booked flight is a '
        "submitted item too). It does NOT apply just because the topic "
        'is IT/hardware-FLAVORED — "get warranty details for my laptop", '
        '"what\'s the warranty on my phone", "check my laptop specs" ask '
        "for information about a personal item itself, not the status of "
        "anything ever submitted, so they do NOT match this rule even "
        'though "get"/"check" are lookup verbs and a laptop is hardware — '
        "these fall through to rule 4 below (a plain IT/hardware "
        'question) instead. Examples, ALL "query_request": "get my request", "show my '
        'request", "see my ticket", "get my request now", "list all the '
        'request", "check the request", "list all my requests", '
        '"what\'s the status of the request", "check my flight status", '
        '"get my flight ticket". '
        "Contrast with rule 5 below: a lookup verb (get/show/see/check/"
        'list/find) never means "create" — only a create verb (raise/'
        "create/open/submit/file) does. Contrast also with a lookup whose "
        "named topic is CLEARLY outside travel/IT — a movie, concert, "
        "restaurant, sporting event, or other personal/entertainment "
        "booking — which does NOT match this rule even though it uses a "
        "lookup verb, because what's being looked up is plainly not a "
        'travel request or helpdesk ticket: "get my movie ticket to '
        'bahubali 2", "show my concert tickets", "check my restaurant '
        'reservation" are NOT "query_request" (they fall through to rule '
        '6, "general_query"). Contrast also with a lookup asking about '
        "general REFERENCE or POLICY information — which countries are "
        "approved/suspended for travel, a holiday calendar, expense "
        "limits, IT policies — rather than a specific request/ticket the "
        'user themselves submitted: "show suspended country", "show '
        'approved country list", "list suspended countries", "what\'s the '
        'holiday calendar" are NOT "query_request" either — there is no '
        "personal record being looked up, just a reference question, so "
        'these also fall through to rule 6, "general_query" (which '
        "routes to helpdesk). Contrast with \"show my travel request to "
        'Cuba" or "check my flight status", which DO name a specific '
        "personal record and stay query_request.\n"
        '3. Does the message contain the word "travel" or "trip", a '
        "destination, or travel dates — ANYWHERE, even alongside a "
        'generic verb like "raise"/"create"/"open" (e.g. "raise a travel '
        'request", "I need a trip to Cuba", "book a flight", "travel '
        'approval")? -> "travel_request". This applies even with zero '
        "other details — an empty travel request is still travel_request, "
        "never ambiguous.\n"
        "4. Does the message name a specific IT/workplace topic — password, "
        'VPN, access, "helpdesk", or "ticket" specifically tied to an '
        'IT/workplace context (e.g. "helpdesk ticket", "IT ticket", '
        '"support ticket", "ticket for my VPN") — even alongside a generic '
        'verb (e.g. "raise a ticket for my VPN", "open a helpdesk '
        'ticket")? -> "helpdesk". A bare "ticket" with no IT/workplace '
        "qualifier anywhere in the message does NOT match this rule on its "
        'own — "book a ticket", "get a movie ticket", "buy a concert '
        'ticket" name no IT/workplace context and must NOT be classified '
        '"helpdesk" (the same principle as rule 2\'s contrastive example, '
        "applied to create-style phrasing instead of lookups).\n"
        "\"Hardware\" here means IT/computing equipment specifically — a "
        "laptop, desktop, monitor, keyboard, mouse, printer, scanner, "
        "phone/mobile device, headset, docking station, or network "
        'equipment ("my laptop screen is flickering", "the printer is out '
        'of paper" -> "helpdesk"). It does NOT mean general office/'
        "facilities equipment just because something is broken or not "
        'working — a coffee machine, a desk chair, air conditioning, '
        "lighting, or furniture are NOT IT hardware, regardless of "
        '"not working"/"broken" phrasing, and must NOT be classified '
        '"helpdesk" on that basis alone: "coffee machine not working", '
        '"my desk chair is broken", "the aircon isn\'t working" are NOT '
        '"helpdesk" (they fall through to rule 6, "general_query" — a '
        "different business team handles facilities/office equipment, "
        "not ICT).\n"
        "A bare greeting or address TO the helpdesk, even with no request "
        'stated yet — "helpdesk", "hi helpdesk", "hello helpdesk team" — '
        "still matches this rule and classifies as \"helpdesk\", never "
        '"general_query", purely because "helpdesk" was named; it does '
        "NOT need an accompanying verb or stated issue the way other "
        "topics under this rule do (a dedicated greeting-message step "
        "handles the reply separately, but classification itself must "
        "still land here).\n"
        "5. Does the message use a CREATE verb — raise, create, open, "
        "submit, file — for a NEW request or ticket that doesn't exist "
        'yet, phrased indefinitely ("a request"/"a ticket", NOT "my '
        'request"/"the request"/"all the requests", which always mean '
        "rule 2 above instead regardless of which verb precedes them) — "
        'and name NO topic at all (e.g. bare "raise a request", "request '
        'needs to be raised", "can you create a request for me", "Can I '
        'create a request for my friend instead of myself?", "Could I '
        'raise a request for someone else?" — a "Can I"/"Could I" polite '
        "phrasing belongs here too, per rule 1's contrast above, not "
        'rule 1)? -> "ambiguous_request" — this is the ONLY case where you ask which '
        "topic they mean, and only because no topic word exists anywhere "
        "in the message, never because details are missing. Excludes "
        'anything containing a "how"/"where" process question (see rule '
        "1's topic-less example) — those match rule 1 instead, even "
        "with no topic named, because asking how/where to do it is a "
        "process question, not an attempt to create something right "
        "now. That exclusion has its own limit, though: a question "
        "asking WHICH topic/category applies to their own request — "
        "not how the mechanics work — still belongs here at rule 5, "
        'not rule 1: "I want to submit a request but I\'m not sure of '
        'the correct category — could you advise?", "not sure if this '
        'is a travel or helpdesk matter, can you help me figure out '
        'which?" -> "ambiguous_request". Contrast with rule 1\'s "how '
        'do I create a travel request" — that asks HOW the mechanics '
        "work for an ALREADY-NAMED feature, not WHICH feature applies "
        "to their situation.\n"
        '6. Otherwise -> "general_query".\n\n'
        "Respond ONLY with a JSON object matching this schema, nothing else:\n"
        "{\n"
        '  "category": "helpdesk" | "travel_request" | "query_request" | '
        '"ambiguous_request" | "general_query" | "cancel_request",\n'
        '  "confidence": number between 0 and 1,\n'
        '  "reasoning": string (one short sentence)\n'
        "}"
    )

    payload = ai_client.build_chat_payload(
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        max_tokens=150,
        response_format={"type": "json_object"},
    )

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:300]}")
        if resp.status_code >= 400:
            # Permanent, config-level failure — bad key, wrong deployment
            # name, malformed request, etc. Retrying changes nothing, so we
            # raise a non-retryable error (skips the retry loop entirely)
            # and log the full detail here since the client won't see it.
            logger.error(
                "Azure OpenAI classification call failed with non-retryable status %s: %s",
                resp.status_code, resp.text[:500],
            )
            raise AzureOpenAINonRetryableError(f"Non-retryable error {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    raw_output = _with_retries(_call)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise AzureOpenAIError(f"Classifier did not return valid JSON: {exc}") from exc

    category = parsed.get("category")
    if category not in ALLOWED_CATEGORIES:
        logger.warning(
            "Classifier returned unrecognized category '%s'; defaulting to 'helpdesk'.",
            category,
        )
        category = "helpdesk"
        parsed["reasoning"] = (parsed.get("reasoning", "") + " (defaulted — unrecognized category)").strip()

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "category": category,
        "confidence": confidence,
        "reasoning": parsed.get("reasoning", ""),
    }


# --------------------------------------------------------------------------
# Route handlers — STUBS to be filled in by the next two build-out steps
# --------------------------------------------------------------------------

HELPDESK_GREETING_MESSAGE = (
    "Hi! \U0001F44B Welcome to the Helpdesk. I'm here to assist you.\n\n"
    "Please let me know how I can help, and briefly explain the issue or "
    "trouble you're facing. I'll be happy to guide you through it."
)


def _is_helpdesk_greeting(text: str, config: dict[str, str]) -> bool:
    """True only when the message just addresses/greets the helpdesk —
    "helpdesk", "hi helpdesk", "hello helpdesk team" — with no actual
    issue, symptom, or question stated yet. A message that names a real
    problem, even briefly (e.g. "helpdesk, my VPN isn't connecting"), is
    NOT a greeting. Travel requests never go through this check — it's
    only consulted for messages already classified as "helpdesk". Fails
    open (treats it as a real question, not a greeting) if the check
    itself errors out, same as every other AI-based check in this
    system — a broken check must never swallow a real issue behind the
    canned welcome message.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "A message was already classified as directed to an IT/workplace "
        "helpdesk. Decide whether it is JUST a greeting or address to the "
        "helpdesk itself, with NO actual issue, symptom, or question "
        'stated yet — e.g. "helpdesk", "hi helpdesk", "hello helpdesk '
        'team", "hey helpdesk, need some help" (still just an opener, no '
        "specific issue named). Tolerate typos and casual phrasing "
        '(e.g. "helpdsk", "hi helpdesk!!"). This is different from a real '
        "request that happens to mention helpdesk but also states an "
        'issue — those are NOT just a greeting: "helpdesk, my VPN isn\'t '
        'connecting", "raise a helpdesk ticket for my printer", "hi, my '
        'laptop won\'t turn on".\n\n'
        'Respond ONLY with a JSON object: {"is_greeting_only": true or false}'
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
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (AzureOpenAIError, requests.RequestException, json.JSONDecodeError) as exc:
        logger.warning("Helpdesk-greeting check failed, treating as a real question: %s", exc)
        return False

    return parsed.get("is_greeting_only") is True


def handle_helpdesk_request(text: str) -> dict[str, Any]:
    """No API call, no search — help-desk questions are just tagged and
    passed through. Whatever consumes this response is responsible for
    actually answering it.
    """
    return {"status": "complete", "question": text}


def handle_general_query(text: str) -> dict[str, Any]:
    """Catch-all for anything that isn't clearly helpdesk/ICT, travel, or a
    lookup of an existing travel request/helpdesk ticket — e.g. "coffee
    machine not working", "get my movie ticket to bahubali 2", "can I get
    reimbursed for my parking". This only ever means the message isn't an
    ICT/travel matter AT ALL — it is NOT a judgment about whether a real
    IT/workplace question happens to be covered by a KB article; this
    codebase never checks against KB articles (there's no KB lookup here
    — that's entirely a downstream system's job once something is tagged
    "helpdesk"), so any message that plausibly names an IT/workplace
    topic still classifies as "helpdesk" and is passed through as normal,
    matching KB article or not. Since 2026-09-22, this returns an
    explicit redirect instead of the silent passthrough used between
    2026-09-02 and then — the previous "no refusal" behavior actively
    forwarded off-topic messages downstream as if they were real ICT
    tickets. No history/lookup logic runs here — that would still need a
    persistence layer and a stable per-user identifier, neither of which
    exist yet.
    """
    return {
        "status": "complete",
        "question": text,
        "message": (
            "Thank you for your enquiry. This matter falls outside the "
            "scope of ICT services. Please contact the relevant business "
            "team responsible for this area, as they will be best placed "
            "to assist you."
        ),
    }


def handle_cancel_request(text: str) -> dict[str, Any]:
    """A fresh (not mid-flow) message expressing intent to cancel/delete/
    withdraw a request or ticket — e.g. "cancel the request", "delete
    request". Since classification only ever runs on a fresh message (an
    in-progress travel/query/ambiguous conversation is intercepted before
    classification and cancelled by its own handler — see
    travel_handler.process_travel_request, query_handler.process_query_request,
    and resolve_ambiguous_request), reaching this handler always means
    there is genuinely nothing in progress to cancel, and this codebase
    has no persistence layer to look up a previously submitted record
    either way.
    """
    return {
        "status": "complete",
        "question": text,
        "message": (
            "You haven't started a travel request or helpdesk ticket yet, "
            "so there's nothing to cancel."
        ),
    }


def handle_travel_request(
    text: str, pending_request: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Single-shot extraction against the travel schema (destination legs
    only — no approver field), with in-code validation and a stateless
    follow-up loop for anything missing.

    If `pending_request` is provided, `text` is treated as the answer to
    the single question that was asked last time — no re-extraction, just
    a direct merge (see travel_handler.process_travel_request).
    """
    try:
        config = _get_config()
        return travel_handler.process_travel_request(text, pending_request, config)
    except travel_handler.TravelExtractionError as exc:
        logger.error("Travel extraction error: %s", exc)
        return {
            "status": "error",
            "message": "Unable to process the travel request right now. Please retry.",
        }
    except country_list.CountryListError as exc:
        logger.error("Approved-country list error: %s", exc)
        return {
            "status": "error",
            "message": "Unable to verify the destination right now. Please retry.",
        }
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return {
            "status": "error",
            "message": "Service is misconfigured. Please contact the administrator.",
        }


def handle_ambiguous_request(text: str) -> dict[str, Any]:
    """Fresh message classified as "wants to raise/create something but
    doesn't say what kind" — asks the user to say which before committing
    to either the travel or helpdesk flow.
    """
    return {
        "status": "incomplete",
        "original_text": text,
        "missing_field": "request_type",
        "follow_up_question": "Is that a travel request or a helpdesk ticket?",
    }


def _classify_ambiguous_request_answer(answer_text: str, config: dict[str, str]) -> Optional[str]:
    """AI-based disambiguation of the answer to "travel request or helpdesk
    ticket?" — a plain keyword match (the original approach here) has zero
    typo tolerance, unlike every other classification in this system (e.g.
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
        'The user was asked: "Is that a travel request or a helpdesk '
        'ticket?" Decide which one their reply means, tolerating typos and '
        "casual phrasing. If the reply doesn't clearly indicate either "
        "one, say so.\n\n"
        'Respond ONLY with a JSON object: {"resolved": "travel_request" | '
        '"helpdesk" | "unclear"}'
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
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (AzureOpenAIError, json.JSONDecodeError) as exc:
        logger.warning("Ambiguous-request disambiguation call failed, treating as unclear: %s", exc)
        return None

    resolved = parsed.get("resolved")
    return resolved if resolved in ("travel_request", "helpdesk") else None


def _is_cancel_intent(text: str, config: dict[str, str]) -> bool:
    """Does this reply express a clear intent to CANCEL/ABORT/STOP the
    in-progress create-request disambiguation entirely — not just decline
    to answer, but end the whole process. Mirrors travel_handler.py's
    same-purpose check. Fails open (treats as NOT a cancellation) if the
    check itself errors out.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        'The user was asked: "Is that a travel request or a helpdesk '
        'ticket?" Decide whether their reply expresses a clear intent to '
        "CANCEL/ABORT/STOP the request entirely — not merely decline to "
        'answer, but actually end the whole process. Words like "cancel", '
        '"close", "exit", "quit", "abort", "never mind", "don\'t bother" '
        "or similar all count, tolerating typos and casual phrasing. A "
        "reply that names travel or helpdesk, even vaguely, does NOT "
        "count.\n\n"
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
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (AzureOpenAIError, json.JSONDecodeError) as exc:
        logger.warning("Cancel-intent check failed, treating as not a cancellation: %s", exc)
        return False

    return parsed.get("is_cancel") is True


def _is_affirmative_restart(text: str, config: dict[str, str]) -> bool:
    """After a travel request is cancelled, the bot's own message asks
    "Let me know if you'd like to start a new one." Decide whether this
    reply is a clear YES to that SPECIFIC offer — "yes", "yeah", "sure",
    "yes please", "ok let's start again", "go ahead" all count. Anything
    else — a different topic, a helpdesk question, "no", or an unrelated
    message — does NOT count, and the caller should classify the message
    fresh instead of forcing a restart. Fails open (treats as NOT
    affirmative) if the check itself errors out — a broken check must
    never force an unwanted restart.
    """
    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "The chatbot just told the user their travel request was "
        'cancelled and asked: "Let me know if you\'d like to start a new '
        'one." Decide whether this reply is a clear YES to that specific '
        'offer — "yes", "yeah", "sure", "yes please", "ok let\'s start '
        'again", "go ahead" all count. Anything else — a different '
        'topic, a helpdesk question, "no", or an unrelated message — '
        "does NOT count.\n\n"
        'Respond ONLY with a JSON object: {"is_affirmative": true or false}'
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
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    try:
        raw_output = _with_retries(_call)
        parsed = json.loads(raw_output)
    except (AzureOpenAIError, json.JSONDecodeError) as exc:
        logger.warning("Affirmative-restart check failed, treating as not affirmative: %s", exc)
        return False

    return parsed.get("is_affirmative") is True


def resolve_ambiguous_request(text: str, pending_request: dict[str, Any], config: dict[str, str]) -> dict[str, Any]:
    """Follow-up: the user has now said travel or helpdesk. Routes into
    that category's normal handler using BOTH the original ambiguous
    message and this disambiguation reply — the reply itself may also
    state real details (e.g. "travel request, I'm going to Cuba for a
    client meeting" answers the question AND gives the destination/
    reason), not just which category it is. Combining them means nothing
    stated in either message has to be repeated.

    Unlike every other follow-up handler, this one returns "belongto"
    itself — which category this resolves to is only known here, whereas
    the caller already knows the category before calling the others.
    """
    if _is_cancel_intent(text, config):
        return {
            "belongto": "ambiguous_request",
            "status": "cancelled",
            "message": "Okay, cancelled. Let me know if you'd like to raise something else.",
        }

    original_text = pending_request.get("original_text", text)
    resolved = _classify_ambiguous_request_answer(text, config)
    combined_text = f"{original_text}. {text}" if text.strip() != original_text.strip() else original_text

    if resolved == "travel_request":
        return {"belongto": "travel_request", **handle_travel_request(combined_text)}
    if resolved == "helpdesk":
        return {"belongto": "helpdesk", **handle_helpdesk_request(combined_text)}

    return {"belongto": "ambiguous_request", **handle_ambiguous_request(original_text)}


def handle_query_request(
    text: str, pending_request: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Classifies whether a lookup is for a travel request or a helpdesk
    ticket and extracts search filters (see query_handler.py). Does not
    perform the actual lookup — there's no persistence layer for either
    record type in this codebase; whatever consumes this response owns
    running the real search against the filters returned here.
    """
    try:
        config = _get_config()
        return query_handler.process_query_request(text, pending_request, config)
    except query_handler.QueryExtractionError as exc:
        logger.error("Query extraction error: %s", exc)
        return {
            "status": "error",
            "message": "Unable to process that lookup right now. Please retry.",
        }
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return {
            "status": "error",
            "message": "Service is misconfigured. Please contact the administrator.",
        }


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _validate_request_body(req: func.HttpRequest) -> tuple[Optional[str], Optional[dict], Optional[str]]:
    """Returns (text, previous_response, error_message). error_message is
    None if valid. `previous_response` is the entire JSON response object
    from the prior call, if the caller is answering a follow-up question —
    None on a fresh message. No server-side session is kept; state travels
    with the request itself.
    """
    try:
        body = req.get_json()
    except ValueError:
        return None, None, "Request body must be valid JSON."

    if not isinstance(body, dict):
        return None, None, "Request body must be a JSON object."

    text = body.get("text")
    if text is None:
        return None, None, "Missing required field 'text'."
    if not isinstance(text, str):
        return None, None, "Field 'text' must be a string."
    text = text.strip()
    if not text:
        return None, None, "Field 'text' must not be empty."
    if len(text) > MAX_INPUT_CHARS:
        return None, None, f"Field 'text' exceeds maximum length of {MAX_INPUT_CHARS} characters."

    previous_response = body.get("previous_response")
    if previous_response is not None and not isinstance(previous_response, dict):
        return None, None, "Field 'previous_response' must be an object if provided."

    return text, previous_response, None


def _error_response(message: str, status_code: int) -> func.HttpResponse:
    payload = {
        "type": None,
        "belongto": None,
        "status": "error",
        "message": message,
    }
    return func.HttpResponse(
        json.dumps(payload), status_code=status_code, mimetype="application/json"
    )


# --------------------------------------------------------------------------
# HTTP trigger
# --------------------------------------------------------------------------

@app.route(route="classify", methods=["POST"])
def classify(req: func.HttpRequest) -> func.HttpResponse:
    logger.info("Received classify/route request.")

    text, previous_response, error = _validate_request_body(req)
    if error:
        logger.info("Input validation failed: %s", error)
        return _error_response(error, status_code=400)

    try:
        config = _get_config()
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return _error_response("Service is misconfigured. Please contact the administrator.", 500)

    # A follow-up answer to a travel request's missing field skips
    # classification entirely — we already know where this is going, and
    # re-classifying a short fragment like "client renewal meeting" risks
    # misrouting it. State travels via `previous_response`, not a session.
    is_travel_followup = (
        previous_response is not None
        and previous_response.get("belongto") == "travel_request"
        and previous_response.get("status") == "incomplete"
    )
    is_query_followup = (
        previous_response is not None
        and previous_response.get("belongto") == "query_request"
        and previous_response.get("status") == "incomplete"
    )
    is_ambiguous_request_followup = (
        previous_response is not None
        and previous_response.get("belongto") == "ambiguous_request"
        and previous_response.get("status") == "incomplete"
    )
    is_travel_cancelled_followup = (
        previous_response is not None
        and previous_response.get("belongto") == "travel_request"
        and previous_response.get("status") == "cancelled"
    )

    try:
        if is_travel_followup:
            handler_result = handle_travel_request(text, pending_request=previous_response)
            response_payload = {"type": "fetch", "belongto": "travel_request", **handler_result}
            logger.info("Continued in-progress travel request.")
            return func.HttpResponse(
                json.dumps(response_payload), status_code=200, mimetype="application/json"
            )

        if is_query_followup:
            handler_result = handle_query_request(text, pending_request=previous_response)
            response_payload = {"type": "fetch", "belongto": "query_request", **handler_result}
            logger.info("Continued in-progress query request.")
            return func.HttpResponse(
                json.dumps(response_payload), status_code=200, mimetype="application/json"
            )

        if is_ambiguous_request_followup:
            handler_result = resolve_ambiguous_request(text, previous_response, config)
            response_payload = {"type": "fetch", **handler_result}
            logger.info("Resolved ambiguous create-request to '%s'.", handler_result.get("belongto"))
            return func.HttpResponse(
                json.dumps(response_payload), status_code=200, mimetype="application/json"
            )

        if is_travel_cancelled_followup and _is_affirmative_restart(text, config):
            # A clear "yes" to the cancelled message's own "let me know if
            # you'd like to start a new one" — reopen a brand new travel
            # request rather than classifying "yes" fresh, which has no
            # topic word and would otherwise fall into general_query.
            # Hardcoded rather than an extraction call: "yes" states no
            # actual trip detail, so this is exactly the same first
            # question travel_handler._validate() returns for an empty leg
            # list.
            handler_result = {
                "status": "incomplete",
                "extracted": [],
                "missing_field_path": "0.reason",
                "follow_up_question": "What's the reason for your trip?",
            }
            response_payload = {"type": "fetch", "belongto": "travel_request", **handler_result}
            logger.info("Restarted a new travel request after a prior cancellation.")
            return func.HttpResponse(
                json.dumps(response_payload), status_code=200, mimetype="application/json"
            )

        classification = _classify_question(text, config)

        if classification["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low-confidence classification (%.2f) for question: %s",
                classification["confidence"], text[:100],
            )

        if classification["category"] == "travel_request":
            handler_result = handle_travel_request(text)
            belongto = "travel_request"
        elif classification["category"] == "query_request":
            handler_result = handle_query_request(text)
            belongto = "query_request"
        elif classification["category"] == "ambiguous_request":
            handler_result = handle_ambiguous_request(text)
            belongto = "ambiguous_request"
        elif classification["category"] == "general_query":
            # Since 2026-09-22: a distinct terminal outcome, not merged
            # into "helpdesk" — see handle_general_query for why. The
            # downstream consumer should treat this the same as a
            # cancelled/complete response: show handler_result["message"]
            # to the user and stop, no ICT ticket handling involved.
            handler_result = handle_general_query(text)
            belongto = "general_query"
        elif classification["category"] == "cancel_request":
            # A cancel/delete-type message with nothing in progress (a
            # mid-flow "cancel" never reaches classification at all — see
            # the three followup checks above and handle_cancel_request's
            # own docstring). Same terminal shape as general_query.
            handler_result = handle_cancel_request(text)
            belongto = "cancel_request"
        else:
            belongto = "helpdesk"
            if _is_helpdesk_greeting(text, config):
                handler_result = {
                    "status": "complete",
                    "question": text,
                    "message": HELPDESK_GREETING_MESSAGE,
                }
            else:
                handler_result = handle_helpdesk_request(text)

        response_payload = {
            "type": "new",
            "belongto": belongto,
            **handler_result,
        }

        logger.info(
            "Classified as '%s' (confidence %.2f), routed to handler.",
            classification["category"], classification["confidence"],
        )
        return func.HttpResponse(
            json.dumps(response_payload), status_code=200, mimetype="application/json"
        )

    except AzureOpenAINonRetryableError as exc:
        logger.error("Azure OpenAI configuration/request error: %s", exc)
        return _error_response(
            "Classification service is misconfigured (check the deployment name, "
            "endpoint, and API key). See server logs for the exact error.",
            500,
        )
    except AzureOpenAIError as exc:
        logger.error("Azure OpenAI error: %s", exc)
        return _error_response("Classification service temporarily unavailable. Please retry.", 502)
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unhandled error during classify/route: %s", exc)
        return _error_response("An unexpected error occurred.", 500)


@app.route(route="helpdesk-search", methods=["POST"])
def helpdesk_search_test(req: func.HttpRequest) -> func.HttpResponse:
    """Isolated test endpoint: calls the help-desk API directly with no
    classification step, and returns both the API's response AND exactly
    what was sent to it (debug_request). Built for the HTML test page —
    use this to verify request correctness before wiring anything else in.
    """
    try:
        body = req.get_json()
    except ValueError:
        return _error_response("Request body must be valid JSON.", 400)

    if not isinstance(body, dict) or not body.get("question"):
        return _error_response("Missing required field 'question'.", 400)

    question = str(body["question"]).strip()
    category = body.get("category")
    previous = body.get("previous")
    if previous is not None and not isinstance(previous, list):
        return _error_response("Field 'previous' must be an array of strings if provided.", 400)

    try:
        result = call_helpdesk_api(question=question, category=category, previous=previous)
        return func.HttpResponse(
            json.dumps({"status": "success", **result}),
            status_code=200,
            mimetype="application/json",
        )
    except HelpdeskAPIConfigError as exc:
        return _error_response(f"Configuration error: {exc}", 500)
    except HelpdeskAPIError as exc:
        return _error_response(f"Help-desk API call failed: {exc}", 502)


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Lightweight liveness check that does not call external dependencies."""
    return func.HttpResponse(
        json.dumps({"status": "healthy"}), status_code=200, mimetype="application/json"
    )
