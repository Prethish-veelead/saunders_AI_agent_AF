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

ALLOWED_CATEGORIES = ["helpdesk", "travel_request", "query_request", "ambiguous_request", "general_query"]

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
        "You classify an incoming chatbot question into exactly one of five "
        "categories. IMPORTANT: completeness is NEVER a reason to pick "
        '"ambiguous_request" or "general_query" — missing details like a '
        "destination, dates, or a specific IT symptom are always resolved "
        "later by follow-up questions, not by classification. The ONLY "
        "question you're answering here is WHICH TOPIC the message is "
        "about (travel vs. IT/helpdesk vs. neither), never whether enough "
        "detail was given.\n\n"
        "Decide in this exact order, stopping at the first match:\n\n"
        "1. Does the message mention looking up / checking / showing an "
        "EXISTING travel request or helpdesk ticket already submitted "
        '(e.g. "show my travel request to Cuba", "show me the ticket '
        'raised on Aug 4", "what\'s the status of my request")? -> '
        '"query_request".\n'
        '2. Does the message contain the word "travel" or "trip", a '
        "destination, or travel dates — ANYWHERE, even alongside a "
        'generic verb like "raise"/"create"/"open" (e.g. "raise a travel '
        'request", "I need a trip to Cuba", "book a flight", "travel '
        'approval")? -> "travel_request". This applies even with zero '
        "other details — an empty travel request is still travel_request, "
        "never ambiguous.\n"
        "3. Does the message name a specific IT/workplace topic (password, "
        'VPN, hardware, access, "ticket", "helpdesk", or similar), even '
        'alongside a generic verb (e.g. "raise a ticket for my VPN", "open '
        'a helpdesk ticket")? -> "helpdesk".\n'
        "4. Does the message ask to CREATE/RAISE/OPEN a request or ticket "
        "but name NO topic at all — no mention of travel/trip and no IT "
        'topic (e.g. bare "raise a request", "request needs to be '
        'raised", "can you create a request for me")? -> '
        '"ambiguous_request" — this is the ONLY case where you ask which '
        "topic they mean, and only because no topic word exists anywhere "
        "in the message, never because details are missing.\n"
        '5. Otherwise -> "general_query".\n\n'
        "Respond ONLY with a JSON object matching this schema, nothing else:\n"
        "{\n"
        '  "category": "helpdesk" | "travel_request" | "query_request" | '
        '"ambiguous_request" | "general_query",\n'
        '  "confidence": number between 0 and 1,\n'
        '  "reasoning": string (one short sentence)\n'
        "}"
    )

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
        "max_tokens": 150,
        "response_format": {"type": "json_object"},
    }

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

def handle_helpdesk_request(text: str) -> dict[str, Any]:
    """No API call, no search — help-desk questions are just tagged and
    passed through. Whatever consumes this response is responsible for
    actually answering it.
    """
    return {"status": "complete", "question": text}


def handle_general_query(text: str) -> dict[str, Any]:
    """Catch-all for anything that isn't clearly helpdesk or travel — e.g.
    "show me what I've asked before". Tagged and passed through only, same
    as helpdesk. No history/lookup logic actually runs here yet — that
    would need a place to persist conversation records (a SharePoint list,
    most likely, given no Azure storage account) plus a stable per-user
    identifier on incoming requests, neither of which exist yet. This stub
    exists so the response contract is defined and testable ahead of that.
    """
    return {"status": "complete", "question": text}


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


def _classify_ambiguous_request_answer(answer_text: str) -> Optional[str]:
    """Deterministic two-way match — no AI call needed for a binary choice
    like this (same pattern as query_handler's record-type disambiguation)."""
    lowered = answer_text.strip().lower()
    if "travel" in lowered or "trip" in lowered:
        return "travel_request"
    if "helpdesk" in lowered or "help desk" in lowered or "help-desk" in lowered or "ticket" in lowered:
        return "helpdesk"
    return None


def resolve_ambiguous_request(text: str, pending_request: dict[str, Any]) -> dict[str, Any]:
    """Follow-up: the user has now said travel or helpdesk. Routes into
    that category's normal handler using the ORIGINAL ambiguous message
    (which carried no extractable info of its own) rather than the
    disambiguation answer itself.

    Unlike every other follow-up handler, this one returns "belongto"
    itself — which category this resolves to is only known here, whereas
    the caller already knows the category before calling the others.
    """
    original_text = pending_request.get("original_text", text)
    resolved = _classify_ambiguous_request_answer(text)

    if resolved == "travel_request":
        return {"belongto": "travel_request", **handle_travel_request(original_text)}
    if resolved == "helpdesk":
        return {"belongto": "helpdesk", **handle_helpdesk_request(original_text)}

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
            handler_result = resolve_ambiguous_request(text, previous_response)
            response_payload = {"type": "fetch", **handler_result}
            logger.info("Resolved ambiguous create-request to '%s'.", handler_result.get("belongto"))
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
        elif classification["category"] == "helpdesk":
            handler_result = handle_helpdesk_request(text)
        elif classification["category"] == "query_request":
            handler_result = handle_query_request(text)
        elif classification["category"] == "ambiguous_request":
            handler_result = handle_ambiguous_request(text)
        else:
            handler_result = handle_general_query(text)

        response_payload = {
            "type": "new",
            "belongto": classification["category"],
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
