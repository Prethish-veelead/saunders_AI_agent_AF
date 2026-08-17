"""
Help-desk retrieval: SharePoint List + Library search.
-------------------------------------------------------

Implements the "search List first, then Library, then give up" logic that
plugs into `handle_helpdesk_request` in function_app.py.

Design decisions baked in (per earlier discussion):
  - No persisted storage anywhere (no Table/Blob storage account available).
    Everything is cached in plain module-level variables, refreshed
    opportunistically when a request arrives and the cache has gone stale.
    List cache: refreshed if older than LIST_CACHE_TTL_SECONDS (default 2h).
    Library cache: refreshed if older than LIBRARY_CACHE_TTL_SECONDS
    (default 4h). This means "check every 2/4 hours" happens on the next
    request after that window elapses, not via a timer.
  - No vector embeddings anywhere. List matching uses fuzzy string matching
    (rapidfuzz) against the Question/Keywords columns. Library matching uses
    simple keyword-overlap scoring against extracted document text.
  - If Azure Functions scales to multiple instances, each instance keeps its
    own cache — slightly more Graph calls than a shared cache, fine at this
    volume (100-1,000 requests/day).
"""

import io
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests
from rapidfuzz import fuzz

logger = logging.getLogger("helpdesk_search")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

AZURE_TENANT_ID = os.environ.get("AZURE_TENANT_ID", "")
SHAREPOINT_CLIENT_ID = os.environ.get("SHAREPOINT_CLIENT_ID", "")
SHAREPOINT_CLIENT_SECRET = os.environ.get("SHAREPOINT_CLIENT_SECRET", "")

# Either provide SHAREPOINT_SITE_ID directly, or HOSTNAME + SITE_PATH so it
# can be resolved once and cached.
SHAREPOINT_SITE_ID = os.environ.get("SHAREPOINT_SITE_ID", "")
SHAREPOINT_HOSTNAME = os.environ.get("SHAREPOINT_HOSTNAME", "")  # e.g. contoso.sharepoint.com
SHAREPOINT_SITE_PATH = os.environ.get("SHAREPOINT_SITE_PATH", "")  # e.g. /sites/Helpdesk

SHAREPOINT_LIST_ID = os.environ.get("SHAREPOINT_LIST_ID", "")

# Column (internal) names on the SharePoint list — adjust to match your list.
LIST_FIELD_QUESTION = os.environ.get("LIST_FIELD_QUESTION", "Title")
LIST_FIELD_ANSWER = os.environ.get("LIST_FIELD_ANSWER", "Answer")
LIST_FIELD_CATEGORY = os.environ.get("LIST_FIELD_CATEGORY", "Category")
LIST_FIELD_KEYWORDS = os.environ.get("LIST_FIELD_KEYWORDS", "Keywords")

LIST_CACHE_TTL_SECONDS = int(os.environ.get("LIST_CACHE_TTL_SECONDS", str(2 * 60 * 60)))
LIBRARY_CACHE_TTL_SECONDS = int(os.environ.get("LIBRARY_CACHE_TTL_SECONDS", str(4 * 60 * 60)))

# NOTE: rapidfuzz scores for genuinely-matching but differently-phrased
# questions typically land in the 40-60 range (not 80-100) — this isn't a
# semantic match, it's character/token overlap. Treat this default as a
# starting point: log match scores in production and tune it against your
# actual list content and real user phrasing.
LIST_MATCH_THRESHOLD = float(os.environ.get("LIST_MATCH_THRESHOLD", "45"))  # rapidfuzz 0-100 scale
LIBRARY_MATCH_MIN_KEYWORD_HITS = int(os.environ.get("LIBRARY_MATCH_MIN_KEYWORD_HITS", "2"))

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))

SUPPORTED_LIBRARY_EXTENSIONS = (".pdf", ".docx")


class SharePointConfigError(Exception):
    """Raised when required SharePoint/Graph configuration is missing."""


class GraphAPIError(Exception):
    """Raised when a Microsoft Graph API call fails unexpectedly."""


def _require_config() -> None:
    missing = []
    if not AZURE_TENANT_ID:
        missing.append("AZURE_TENANT_ID")
    if not SHAREPOINT_CLIENT_ID:
        missing.append("SHAREPOINT_CLIENT_ID")
    if not SHAREPOINT_CLIENT_SECRET:
        missing.append("SHAREPOINT_CLIENT_SECRET")
    if not SHAREPOINT_SITE_ID and not (SHAREPOINT_HOSTNAME and SHAREPOINT_SITE_PATH):
        missing.append("SHAREPOINT_SITE_ID (or SHAREPOINT_HOSTNAME + SHAREPOINT_SITE_PATH)")
    if not SHAREPOINT_LIST_ID:
        missing.append("SHAREPOINT_LIST_ID")
    if missing:
        raise SharePointConfigError(f"Missing required configuration: {', '.join(missing)}")


# --------------------------------------------------------------------------
# Auth: client-credentials token, cached until near expiry
# --------------------------------------------------------------------------

@dataclass
class _TokenCache:
    access_token: Optional[str] = None
    expires_at: float = 0.0


_token_cache = _TokenCache()


def _get_graph_token() -> str:
    now = time.time()
    if _token_cache.access_token and now < (_token_cache.expires_at - 60):
        return _token_cache.access_token

    url = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token"
    data = {
        "grant_type": "client_credentials",
        "client_id": SHAREPOINT_CLIENT_ID,
        "client_secret": SHAREPOINT_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    }
    resp = requests.post(url, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise GraphAPIError(f"Failed to acquire Graph token: {resp.status_code} {resp.text[:200]}")

    body = resp.json()
    _token_cache.access_token = body["access_token"]
    _token_cache.expires_at = now + int(body.get("expires_in", 3600))
    logger.info("Refreshed Microsoft Graph access token.")
    return _token_cache.access_token


def _graph_get(path: str, params: Optional[dict[str, str]] = None) -> dict[str, Any]:
    token = _get_graph_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{GRAPH_BASE_URL}{path}", headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS
    )
    if resp.status_code != 200:
        raise GraphAPIError(f"Graph GET {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()


def _graph_get_bytes(path: str) -> bytes:
    token = _get_graph_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{GRAPH_BASE_URL}{path}", headers=headers, timeout=REQUEST_TIMEOUT_SECONDS)
    if resp.status_code != 200:
        raise GraphAPIError(f"Graph GET {path} failed: {resp.status_code} {resp.text[:200]}")
    return resp.content


# --------------------------------------------------------------------------
# Site ID resolution (cached indefinitely — a site's ID never changes)
# --------------------------------------------------------------------------

_resolved_site_id: Optional[str] = None


def _get_site_id() -> str:
    global _resolved_site_id
    if SHAREPOINT_SITE_ID:
        return SHAREPOINT_SITE_ID
    if _resolved_site_id:
        return _resolved_site_id

    site_data = _graph_get(f"/sites/{SHAREPOINT_HOSTNAME}:{SHAREPOINT_SITE_PATH}")
    _resolved_site_id = site_data["id"]
    logger.info("Resolved SharePoint site id.")
    return _resolved_site_id


# --------------------------------------------------------------------------
# List cache
# --------------------------------------------------------------------------

@dataclass
class _ListCache:
    items: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0


_list_cache = _ListCache()


def _fetch_list_items() -> list[dict[str, Any]]:
    site_id = _get_site_id()
    data = _graph_get(
        f"/sites/{site_id}/lists/{SHAREPOINT_LIST_ID}/items",
        params={"$expand": "fields"},
    )
    items = []
    for entry in data.get("value", []):
        fields_data = entry.get("fields", {})
        question = fields_data.get(LIST_FIELD_QUESTION, "")
        answer = fields_data.get(LIST_FIELD_ANSWER, "")
        if not question or not answer:
            continue  # skip unanswered tickets — nothing to retrieve yet
        items.append({
            "question": question,
            "answer": answer,
            "category": fields_data.get(LIST_FIELD_CATEGORY, ""),
            "keywords": fields_data.get(LIST_FIELD_KEYWORDS, ""),
            "item_id": entry.get("id"),
        })
    return items


def _get_cached_list_items() -> list[dict[str, Any]]:
    now = time.time()
    if _list_cache.items and (now - _list_cache.fetched_at) < LIST_CACHE_TTL_SECONDS:
        return _list_cache.items

    try:
        _list_cache.items = _fetch_list_items()
        _list_cache.fetched_at = now
        logger.info("Refreshed list cache: %d answered entries.", len(_list_cache.items))
    except GraphAPIError as exc:
        if _list_cache.items:
            logger.warning("List refresh failed, serving stale cache: %s", exc)
        else:
            raise
    return _list_cache.items


# --------------------------------------------------------------------------
# Library cache (extracted document text)
# --------------------------------------------------------------------------

@dataclass
class _LibraryCache:
    documents: list[dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0


_library_cache = _LibraryCache()


def _extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()
    try:
        if lower.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        if lower.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to extract text from '%s': %s", filename, exc)
        return ""
    return ""


def _fetch_library_documents() -> list[dict[str, Any]]:
    site_id = _get_site_id()
    data = _graph_get(f"/sites/{site_id}/drive/root/children")

    documents = []
    for entry in data.get("value", []):
        name = entry.get("name", "")
        if not name.lower().endswith(SUPPORTED_LIBRARY_EXTENSIONS):
            continue
        item_id = entry.get("id")
        try:
            content = _graph_get_bytes(f"/sites/{site_id}/drive/items/{item_id}/content")
            text = _extract_text(name, content)
        except GraphAPIError as exc:
            logger.warning("Skipping '%s' — download failed: %s", name, exc)
            continue
        if text.strip():
            documents.append({
                "title": name,
                "text": text,
                "item_id": item_id,
                "last_modified": entry.get("lastModifiedDateTime", ""),
            })
    return documents


def _get_cached_library_documents() -> list[dict[str, Any]]:
    now = time.time()
    if _library_cache.documents and (now - _library_cache.fetched_at) < LIBRARY_CACHE_TTL_SECONDS:
        return _library_cache.documents

    try:
        _library_cache.documents = _fetch_library_documents()
        _library_cache.fetched_at = now
        logger.info("Refreshed library cache: %d documents.", len(_library_cache.documents))
    except GraphAPIError as exc:
        if _library_cache.documents:
            logger.warning("Library refresh failed, serving stale cache: %s", exc)
        else:
            raise
    return _library_cache.documents


# --------------------------------------------------------------------------
# Matching logic (no vectors — fuzzy string match + keyword overlap)
# --------------------------------------------------------------------------

def _match_against_list(question: str, items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    best_item = None
    best_score = 0.0
    for item in items:
        candidate_text = f"{item['question']} {item['keywords']}"
        score = fuzz.token_set_ratio(question, candidate_text)
        if score > best_score:
            best_score = score
            best_item = item

    if best_item and best_score >= LIST_MATCH_THRESHOLD:
        logger.info("List match found (score %.1f): %s", best_score, best_item["question"][:80])
        return {"source": "list", "score": best_score, **best_item}
    return None


def _match_against_library(question: str, documents: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    question_words = {
        cleaned for w in question.split()
        if len(cleaned := re.sub(r"[^\w]", "", w).lower()) > 2
    }
    if not question_words:
        return None

    best_doc = None
    best_hits = 0
    best_snippet = ""

    for doc in documents:
        text = doc["text"]
        text_lower = text.lower()
        hits = sum(1 for w in question_words if w in text_lower)
        if hits > best_hits:
            best_hits = hits
            best_doc = doc
            # Grab a window of text around the first matching keyword as a snippet,
            # rather than passing the whole document to the model.
            first_match_pos = next(
                (text_lower.find(w) for w in question_words if w in text_lower), 0
            )
            start = max(0, first_match_pos - 300)
            end = min(len(text), first_match_pos + 700)
            best_snippet = text[start:end]

    if best_doc and best_hits >= LIBRARY_MATCH_MIN_KEYWORD_HITS:
        logger.info("Library match found (%d keyword hits): %s", best_hits, best_doc["title"])
        return {"source": "library", "title": best_doc["title"], "snippet": best_snippet, "hits": best_hits}
    return None


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def find_helpdesk_context(question: str) -> Optional[dict[str, Any]]:
    """Searches the SharePoint List first, then the Library.

    Returns a dict describing the matched context (source + text), or None
    if nothing relevant was found in either location — the caller should
    treat None as "kick off the ticket-raising flow."
    """
    _require_config()

    list_items = _get_cached_list_items()
    list_match = _match_against_list(question, list_items)
    if list_match:
        return list_match

    library_docs = _get_cached_library_documents()
    library_match = _match_against_library(question, library_docs)
    if library_match:
        return library_match

    logger.info("No match found in List or Library for question: %s", question[:100])
    return None
