"""
Help-desk retrieval: third-party search API client.
-----------------------------------------------------

Replaces the earlier SharePoint List/Library search — the help-desk answer
now comes entirely from an existing external API. This module's only job is
to call it correctly and surface exactly what was sent, so the calling
Function (and the HTML test page) can verify the request shape.
"""

import logging
import os
import time
from typing import Any, Optional

import requests

logger = logging.getLogger("helpdesk_api_client")

HELPDESK_API_BASE = os.environ.get("HELPDESK_API_BASE", "")
HELPDESK_API_KEY = os.environ.get("HELPDESK_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_BASE_DELAY_SECONDS = float(os.environ.get("RETRY_BASE_DELAY_SECONDS", "1.5"))


class HelpdeskAPIConfigError(Exception):
    """Raised when HELPDESK_API_BASE / HELPDESK_API_KEY are not configured."""


class HelpdeskAPIError(Exception):
    """Raised when the third-party API call ultimately fails."""


def _require_config() -> None:
    if not HELPDESK_API_BASE:
        raise HelpdeskAPIConfigError("HELPDESK_API_BASE is not configured.")
    if not HELPDESK_API_KEY:
        raise HelpdeskAPIConfigError("HELPDESK_API_KEY is not configured.")


def call_helpdesk_api(
    question: str,
    category: Optional[str] = None,
    previous: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Calls {API_BASE}/search.json and returns a dict with the parsed
    response plus a 'debug_request' block describing exactly what was sent
    (API key masked) — this is what the HTML tester displays so you can
    confirm the request is shaped correctly.
    """
    _require_config()

    url = f"{HELPDESK_API_BASE.rstrip('/')}/search.json"
    params: dict[str, str] = {"q": question}
    if category:
        params["category"] = category
    if previous:
        # Assumption: comma-separated string of prior questions. Adjust here
        # if the real API expects a different shape (repeated params, JSON
        # array, etc.) — this is the one place that would need to change.
        params["previous"] = ",".join(previous)

    headers = {"x-api-key": HELPDESK_API_KEY}

    debug_request = {
        "method": "GET",
        "url": url,
        "params": params,
        "headers": {"x-api-key": _mask(HELPDESK_API_KEY)},
    }

    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise HelpdeskAPIError(f"Transient error {resp.status_code}: {resp.text[:200]}")
            if resp.status_code != 200:
                raise HelpdeskAPIError(f"API returned {resp.status_code}: {resp.text[:200]}")
            return {"response": resp.json(), "debug_request": debug_request}
        except (requests.RequestException, HelpdeskAPIError) as exc:
            last_exc = exc
            if attempt == MAX_RETRIES:
                break
            delay = RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
            logger.warning(
                "Help-desk API call failed (attempt %d/%d): %s. Retrying in %.1fs.",
                attempt, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)

    raise HelpdeskAPIError(f"Help-desk API call failed after {MAX_RETRIES} attempts") from last_exc


def _mask(secret: str) -> str:
    if len(secret) <= 4:
        return "***"
    return secret[:2] + "***" + secret[-2:]
