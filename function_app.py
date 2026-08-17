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

from helpdesk_search import find_helpdesk_context, SharePointConfigError, GraphAPIError

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

ALLOWED_CATEGORIES = ["helpdesk", "travel_request"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("classify_router")

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""


class AzureOpenAIError(Exception):
    """Raised when the Azure OpenAI API call ultimately fails."""


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
        "You classify an incoming chatbot question into exactly one of two "
        'categories: "helpdesk" (IT support, password resets, hardware, '
        'access requests, general workplace questions) or "travel_request" '
        "(booking flights/hotels, travel approvals, itineraries, "
        "reimbursement for trips). "
        "Respond ONLY with a JSON object matching this schema, nothing else:\n"
        "{\n"
        '  "category": "helpdesk" | "travel_request",\n'
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
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
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

def _generate_grounded_answer(question: str, context: dict[str, Any], config: dict[str, str]) -> str:
    """Asks gpt-4o-mini to answer using only the retrieved context. Returns
    the model's answer text, or the literal string "NOT_FOUND" if the model
    decides the context doesn't actually address the question.
    """
    if context["source"] == "list":
        reference_text = context["answer"]
    else:
        reference_text = context["snippet"]

    url = (
        f"{config['endpoint'].rstrip('/')}/openai/deployments/"
        f"{AZURE_OPENAI_CHAT_DEPLOYMENT}/chat/completions"
        f"?api-version={AZURE_OPENAI_API_VERSION}"
    )
    headers = {"api-key": config["api_key"], "Content-Type": "application/json"}
    system_prompt = (
        "Answer the user's question using ONLY the reference content provided. "
        "Keep the answer short and direct. If the reference content does not "
        "actually answer the question, respond with exactly: NOT_FOUND"
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Reference content:\n{reference_text}\n\nQuestion:\n{question}"},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }

    def _call():
        resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code >= 500 or resp.status_code == 429:
            raise AzureOpenAIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    return _with_retries(_call)


def handle_helpdesk_request(text: str, classification: dict[str, Any]) -> dict[str, Any]:
    """Searches the SharePoint List first, then the Library, and generates a
    grounded answer from whichever match is found. If neither location has
    a relevant answer, signals that the ticket-raising flow should start.

    TODO: once the helpdesk_v1 schema-driven conversation is built, call it
    here instead of returning the "needs_ticket" placeholder below.
    """
    try:
        config = _get_config()
        context = find_helpdesk_context(text)

        if context is None:
            return {
                "handler": "helpdesk",
                "status": "needs_ticket",
                "message": "No matching answer found in SharePoint Lists or Libraries.",
            }

        answer = _generate_grounded_answer(text, context, config)

        if answer == "NOT_FOUND":
            logger.info("Retrieved context did not actually answer the question; escalating to ticket.")
            return {
                "handler": "helpdesk",
                "status": "needs_ticket",
                "message": "Closest match did not sufficiently answer the question.",
            }

        return {
            "handler": "helpdesk",
            "status": "answered",
            "answer": answer,
            "source": context["source"],
            "source_detail": context.get("question") if context["source"] == "list" else context.get("title"),
        }

    except SharePointConfigError as exc:
        logger.error("SharePoint configuration error: %s", exc)
        return {
            "handler": "helpdesk",
            "status": "error",
            "message": "Help-desk search is misconfigured. Please contact the administrator.",
        }
    except GraphAPIError as exc:
        logger.error("Graph API error during helpdesk search: %s", exc)
        return {
            "handler": "helpdesk",
            "status": "error",
            "message": "Unable to search SharePoint right now. Please retry shortly.",
        }
    except AzureOpenAIError as exc:
        logger.error("Azure OpenAI error while generating helpdesk answer: %s", exc)
        return {
            "handler": "helpdesk",
            "status": "error",
            "message": "Answer generation temporarily unavailable. Please retry.",
        }


def handle_travel_request(text: str, classification: dict[str, Any]) -> dict[str, Any]:
    """TODO: plug in the travel_v1 schema-driven intake conversation here.

    Expected eventual behavior:
      1. Load the travel_v1 schema.
      2. Walk the user through each field conversationally (approver email,
         then the repeatable "legs" group for each destination).
      3. Validate as fields come in (e.g. endDate not before startDate).
      4. Submit the completed request once all required fields are collected.

    For now this returns a clearly-marked placeholder so the classify/route
    path can be tested end-to-end before that logic exists.
    """
    logger.info("Routed to travel handler (stub). Question: %s", text[:100])
    return {
        "handler": "travel_request",
        "status": "pending_implementation",
        "message": (
            "Travel handler not yet implemented — this request would start the "
            "travel_v1 schema-driven intake flow."
        ),
    }


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------

def _validate_request_body(req: func.HttpRequest) -> tuple[Optional[str], Optional[str]]:
    """Returns (text, error_message). error_message is None if valid."""
    try:
        body = req.get_json()
    except ValueError:
        return None, "Request body must be valid JSON."

    if not isinstance(body, dict):
        return None, "Request body must be a JSON object."

    text = body.get("text")
    if text is None:
        return None, "Missing required field 'text'."
    if not isinstance(text, str):
        return None, "Field 'text' must be a string."
    text = text.strip()
    if not text:
        return None, "Field 'text' must not be empty."
    if len(text) > MAX_INPUT_CHARS:
        return None, f"Field 'text' exceeds maximum length of {MAX_INPUT_CHARS} characters."

    return text, None


def _error_response(message: str, status_code: int) -> func.HttpResponse:
    payload = {
        "classified_category": None,
        "confidence": 0.0,
        "reasoning": message,
        "status": "error",
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

    text, error = _validate_request_body(req)
    if error:
        logger.info("Input validation failed: %s", error)
        return _error_response(error, status_code=400)

    try:
        config = _get_config()
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        return _error_response("Service is misconfigured. Please contact the administrator.", 500)

    try:
        classification = _classify_question(text, config)

        if classification["confidence"] < LOW_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low-confidence classification (%.2f) for question: %s",
                classification["confidence"], text[:100],
            )

        if classification["category"] == "travel_request":
            handler_result = handle_travel_request(text, classification)
        else:
            handler_result = handle_helpdesk_request(text, classification)

        response_payload = {
            "classified_category": classification["category"],
            "confidence": classification["confidence"],
            "reasoning": classification["reasoning"],
            "status": "success",
            "handler_result": handler_result,
        }

        logger.info(
            "Classified as '%s' (confidence %.2f), routed to handler.",
            classification["category"], classification["confidence"],
        )
        return func.HttpResponse(
            json.dumps(response_payload), status_code=200, mimetype="application/json"
        )

    except AzureOpenAIError as exc:
        logger.error("Azure OpenAI error: %s", exc)
        return _error_response("Classification service temporarily unavailable. Please retry.", 502)
    except Exception as exc:  # noqa: BLE001 - top-level safety net
        logger.exception("Unhandled error during classify/route: %s", exc)
        return _error_response("An unexpected error occurred.", 500)


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Lightweight liveness check that does not call external dependencies."""
    return func.HttpResponse(
        json.dumps({"status": "healthy"}), status_code=200, mimetype="application/json"
    )
