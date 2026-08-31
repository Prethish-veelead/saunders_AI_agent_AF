"""
Approved travel destination list.
-----------------------------------

Fetches the "CountryMaster" SharePoint list via Microsoft Graph. Two
different auth mechanisms are supported, auto-selected from whichever
settings are configured — the two deployments genuinely need different
ones, not just different values of the same setting:

  - Client side: the Function App's own Managed Identity
    (`DefaultAzureCredential`, no secret anywhere). Works because the
    Function App and the client's SharePoint site live in the SAME Entra
    ID tenant — a managed identity cannot be granted access to a
    resource in a different tenant, full stop.
  - Our (VeeLead) side: `AZURE_TENANT_ID` / `SHAREPOINT_CLIENT_ID` /
    `SHAREPOINT_CLIENT_SECRET`, the client-credentials flow — required
    here specifically because `saunders-ai-agent-func` (tenant
    dada034a-...) and `veeleaddev.sharepoint.com` (tenant 08a7d6c3-...)
    are in two different Entra ID tenants, which is exactly the case a
    managed identity can't reach. Same pattern already proven in the
    now-unused helpdesk_search.py from earlier in this project.

If `AZURE_TENANT_ID`/`SHAREPOINT_CLIENT_ID`/`SHAREPOINT_CLIENT_SECRET`
are all set, that takes priority (cross-tenant case); otherwise this
falls back to the managed identity (same-tenant case).

Design, otherwise unrelated to which auth path is used:
  1. The two deployments point at two different SharePoint sites/lists,
     and even the column holding the country name has a different
     display name between them ("Location Name" vs. "Country Name") —
     all of that is Function App configuration, not hardcoded.
  2. The column's real (internal) name is resolved once from its display
     name and cached alongside the country values — Graph returns list
     item fields keyed by internal name, which SharePoint often mangles
     from the display name, so this can't be assumed or hardcoded.
  3. Cached with a TTL: the list rarely changes, and this avoids a Graph
     round trip on every travel request. Checked with `is not None`, not
     truthiness — an empty list is still a valid (if unlikely) cached
     result and must not defeat the TTL and refetch every call.
  4. A failure to fetch is treated as a system error, not "allow
     anything" — the whole point of this list is to enforce which
     countries are permitted, so silently bypassing the check on a Graph
     hiccup would defeat it.
"""

import logging
import os
import time
from typing import Any, Optional

import requests
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("country_list")

AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID", "")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET", "")

SHAREPOINT_HOSTNAME = os.environ.get("SHAREPOINT_HOSTNAME", "")
SHAREPOINT_SITE_PATH = os.environ.get("SHAREPOINT_SITE_PATH", "")
SHAREPOINT_COUNTRY_LIST_ID = os.environ.get("SHAREPOINT_COUNTRY_LIST_ID", "")
SHAREPOINT_COUNTRY_COLUMN_DISPLAY_NAME = os.environ.get("SHAREPOINT_COUNTRY_COLUMN_DISPLAY_NAME", "Location Name")
COUNTRY_LIST_CACHE_TTL_SECONDS = int(os.environ.get("COUNTRY_LIST_CACHE_TTL_SECONDS", "3600"))
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))

_USE_CLIENT_CREDENTIALS = bool(AZURE_TENANT_ID and SHAREPOINT_CLIENT_ID and SHAREPOINT_CLIENT_SECRET)

_credential: Optional[DefaultAzureCredential] = None
_client_cred_token: dict[str, Any] = {"access_token": None, "expires_at": 0.0}
_cache: dict[str, Any] = {"countries": None, "column_name": None, "fetched_at": 0.0}


class CountryListError(Exception):
    """Raised when the approved-country list can't be fetched or is
    misconfigured — treated as a system error, never as "allow it"."""


def _get_client_credentials_token() -> str:
    now = time.time()
    if _client_cred_token["access_token"] and now < (_client_cred_token["expires_at"] - 60):
        return _client_cred_token["access_token"]

    url = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": SHAREPOINT_CLIENT_ID,
        "client_secret": SHAREPOINT_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(url, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise CountryListError(f"Failed to acquire Graph token: {resp.status_code} {resp.text[:200]}")

    body = resp.json()
    _client_cred_token["access_token"] = body["access_token"]
    _client_cred_token["expires_at"] = now + int(body.get("expires_in", 3600))
    return _client_cred_token["access_token"]


def _get_managed_identity_token() -> str:
    global _credential
    if _credential is None:
        _credential = DefaultAzureCredential()
    try:
        token = _credential.get_token("https://graph.microsoft.com/.default")
    except Exception as exc:  # covers azure.core.exceptions.ClientAuthenticationError and friends
        raise CountryListError(f"Could not acquire a Graph token via managed identity: {exc}") from exc
    return token.token


def _get_graph_headers() -> dict[str, str]:
    token = _get_client_credentials_token() if _USE_CLIENT_CREDENTIALS else _get_managed_identity_token()
    return {"Authorization": f"Bearer {token}"}


def _require_config() -> None:
    if not SHAREPOINT_HOSTNAME or not SHAREPOINT_SITE_PATH or not SHAREPOINT_COUNTRY_LIST_ID:
        raise CountryListError(
            "SHAREPOINT_HOSTNAME, SHAREPOINT_SITE_PATH, and SHAREPOINT_COUNTRY_LIST_ID must all be configured."
        )


def _site_list_base_url() -> str:
    site_path = SHAREPOINT_SITE_PATH.strip("/")
    return f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_HOSTNAME}:/{site_path}:/lists/{SHAREPOINT_COUNTRY_LIST_ID}"


def _resolve_column_internal_name(headers: dict[str, str]) -> str:
    url = f"{_site_list_base_url()}/columns"
    resp = requests.get(
        url, headers=headers, params={"$select": "name,displayName"}, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if resp.status_code != 200:
        raise CountryListError(f"Could not read CountryMaster columns: {resp.status_code} {resp.text[:200]}")
    for col in resp.json().get("value", []):
        if col.get("displayName") == SHAREPOINT_COUNTRY_COLUMN_DISPLAY_NAME:
            return col["name"]
    raise CountryListError(
        f"No column named '{SHAREPOINT_COUNTRY_COLUMN_DISPLAY_NAME}' found in the CountryMaster list."
    )


def _fetch_countries() -> tuple[list[str], str]:
    _require_config()
    headers = _get_graph_headers()
    column_name = _resolve_column_internal_name(headers)

    countries: list[str] = []
    url = f"{_site_list_base_url()}/items"
    params: Optional[dict[str, str]] = {"$expand": "fields", "$top": "200"}

    while url:
        resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if resp.status_code != 200:
            raise CountryListError(f"Could not read CountryMaster items: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        for item in data.get("value", []):
            value = item.get("fields", {}).get(column_name)
            if value:
                countries.append(str(value).strip())
        url = data.get("@odata.nextLink")
        params = None  # nextLink already carries the query string

    return countries, column_name


def get_approved_countries() -> list[str]:
    """Returns the cached list of approved destination country names,
    refetching from SharePoint if the cache is empty (never fetched) or
    has expired. Raises CountryListError if the fetch fails — callers
    must not treat that as "no restriction."
    """
    now = time.time()
    if _cache["countries"] is not None and (now - _cache["fetched_at"]) < COUNTRY_LIST_CACHE_TTL_SECONDS:
        return _cache["countries"]

    countries, column_name = _fetch_countries()
    _cache["countries"] = countries
    _cache["column_name"] = column_name
    _cache["fetched_at"] = now
    logger.info("Refreshed approved country list: %d entries (column '%s').", len(countries), column_name)
    return countries


def is_country_approved(country: str) -> bool:
    """Case-insensitive membership check against the approved list."""
    approved = get_approved_countries()
    lowered = country.strip().lower()
    return any(lowered == c.strip().lower() for c in approved)
