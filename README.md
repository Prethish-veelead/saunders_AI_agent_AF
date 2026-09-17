# Router Function — Chatbot Classify & Route API

Azure Function backend for a chatbot. It reads an incoming message, figures out what the person actually wants, and routes it to the right flow — creating a travel request, opening a helpdesk question, looking up something already submitted, or asking a clarifying question when the message alone isn't enough to tell.

This document describes **what the system does today**. It is kept up to date as the client-facing behavior changes — see the [Changelog](#changelog) at the bottom for a dated history of what changed and why.

---

## What it does

A single HTTP endpoint (`/api/classify`) accepts a free-text message and returns one of five outcomes:

| Category | What it means | Example message |
|---|---|---|
| `travel_request` | Create a new trip | *"Business trip to Cuba with client, 26 Aug 2026 to 24 Sep 2026"* |
| `helpdesk` | A new IT/workplace question — also anything unrelated to travel/helpdesk (see note below) | *"How do I reset my VPN password?"*, *"get my movie ticket to bahubali 2"* |
| `query_request` | Look up an existing travel request or helpdesk ticket | *"show my travel request to Cuba"*, *"list all the requests"* |
| `ambiguous_request` | Wants to create/raise something, but doesn't say what kind | *"raise a request"* |

> `general_query` — the catch-all for anything unrelated to travel/helpdesk — is still detected internally by the classifier, but as of 2026-09-02 is routed to the `helpdesk` handler rather than surfaced as its own outcome (see Changelog). It never appears in a live response's `belongto`.

Classification is done by an Azure OpenAI (gpt-4o-mini) call. Everything after that — extracting the right details, validating them, deciding what to ask next — is deterministic code, not left to the model to improvise.

---

## How a conversation works

The API is **stateless** — the Function App itself remembers nothing between calls. Continuing a conversation is done by the caller (the chatbot frontend) echoing state back on each request. Every response includes a `status`:

- **`"complete"`** — nothing more needed.
- **`"incomplete"`** — the bot needs an answer to `follow_up_question` before it can finish.
- **`"error"`** — something failed.
- **`"cancelled"`** (`travel_request` only) — the user asked to cancel a request that was in progress; `message` explains it.

**The contract for a caller:**
1. When `status` comes back `"incomplete"`, show the user `follow_up_question` and save the **entire response object**.
2. When the user replies, send their new message as `text`, and send the saved object back as `previous_response`, unmodified:
   ```json
   { "text": "the user's reply", "previous_response": { ...the full previous response... } }
   ```
3. When `status` comes back `"complete"`, `"error"`, or `"cancelled"`, clear the saved state — the next message starts fresh.

Any client integrating with this API should check `status === "incomplete"` **generically** to decide whether to keep the conversation state — not check for one specific category. Three of the five categories currently use follow-up questions (`travel_request`, `query_request`, `ambiguous_request`), and hardcoding to just one of them silently breaks continuation for the others.

---

## The five categories, in detail

### `travel_request` — creating a trip

Extracts one or more trip "legs" (destination, dates, reason, and — only when relevant — who the trip is for) from the message in a single AI call, including multiple destinations mentioned in one message. Anything missing gets asked about **one field at a time**, always in this fixed order: **requestFor (only if triggered) → reason → destination → start date → end date**.

Single message, everything given:
```json
{
  "type": "new", "belongto": "travel_request", "status": "complete",
  "extracted": [
    { "destinationCountry": "Cuba", "startDate": "2027-08-05", "endDate": "2027-08-06", "reason": "client meeting", "requestFor": null }
  ]
}
```

**`requestFor`** is `null` for the ordinary case — the requester is traveling themselves — and nothing about the flow changes: no extra field, no extra question, identical to before this existed. It's only populated when the conversation itself signals the trip is for someone else — "for my manager", "on behalf of my friend", "for John" — and this is recognized wherever it's mentioned, not just as the answer to a dedicated question (e.g. stating it while answering the destination question also works: *"cuba, not me, for my friend"* captures both `destinationCountry` and `requestFor` from the same reply).

An **email address is always required** to consider this resolved — a bare placeholder ("someone", "my manager", "him") gets the broad question, and a name alone ("John") isn't treated as sufficient either, since it can't uniquely identify who to book the trip for:
```json
{
  "extracted": [ { "destinationCountry": null, "startDate": null, "endDate": null, "reason": null, "requestFor": "my manager" } ],
  "missing_field_path": "0.requestFor",
  "follow_up_question": "Who is this travel request for? Could you share their name or email address?"
}
```
Once a name is known but no email yet, the follow-up narrows instead of repeating itself:
```json
{
  "extracted": [ { "destinationCountry": null, "startDate": null, "endDate": null, "reason": null, "requestFor": "John" } ],
  "missing_field_path": "0.requestFor",
  "follow_up_question": "Could you share John's email address?"
}
```
An email given anywhere — including inline in the very first message ("for john@company.com") — is accepted immediately with no question at all, and combines with an already-known name (`"John"` + `"john@company.com"` → `"John, john@company.com"`).

**`startDate`/`endDate` only ever need the date** — `"YYYY-MM-DD"` — time is not collected in this conversation (dropped 2026-09-11: in practice users don't give it when asked, so asking just added friction with no benefit). If a message happens to volunteer a time anyway (e.g. *"5 Aug 2027 9:30am"*), it's still captured as `"YYYY-MM-DDTHH:mm"` rather than discarded, but a date with no time is just as complete — never blocked or re-asked over a missing time. A bare time with nothing to attach it to (e.g. answering the date question with just *"9:30am"*) is never accepted as if it were a date, and never silently defaults to today's date — the question is asked again instead.

Missing details, ordinary self-request — follow-up question:
```json
{
  "type": "new", "belongto": "travel_request", "status": "incomplete",
  "extracted": [],
  "missing_field_path": "0.reason",
  "follow_up_question": "What's the reason for your trip?"
}
```

A message can also describe several trips at once — each becomes its own entry in `extracted`, resolved in order (leg 1 must be complete before leg 2 is asked about; a second or later leg's questions name its destination explicitly, e.g. *"What's the start date for the trip to Bangladesh?"*, so a multi-destination itinerary can't be answered against the wrong leg). **Exception:** if every leg already has a valid destination and dates and 2+ of them are only missing the reason, that's asked **once** for all of them (`missing_field_path: "*.reason"`, *"What's the reason for the trips?"*) and the single answer is applied to every leg still missing it — not asked per leg.

**Destination is checked against an approved list**, not accepted freely — see [Approved destination list](#approved-destination-list) below. A well-formed but unapproved country is rejected the same way a malformed one is:
```json
{
  "follow_up_question": "Sorry, Vatican City isn't a country we currently support for travel requests — could you confirm a different destination?"
}
```

**Reliability details worth knowing:**
- Date/time phrasing is normalized regardless of format (`26 Aug 2026`, `26-Aug-2026`, `2026-08-26`, `9:30am`, `half nine`, `6 in the evening` all work). Ambiguous all-numeric dates (`7-9-2026`) are read as day-first, not US month-first.
- A follow-up answer that gives more than the one thing asked (e.g. destination *and* both start/end in one reply) is still fully captured — a scoped AI extraction pulls everything relevant from the answer instead of only using the single field that was asked about, and does so for every follow-up field (including date/time) rather than trusting a library parser, since a bare time-only reply could otherwise be misread as a brand new date.
- Sanity checks run in code, not just trusted from the AI: a start date can't be in the past, an end date can't be before the start date, and a destination field that contains digits is rejected and re-asked (the AI occasionally folds a date into the destination field by mistake — this catches it every time).

### `helpdesk` — a new IT/workplace question (also the catch-all)

No extraction, no follow-up — the question is tagged and passed straight through. Answering it is owned by whatever consumes this response (a separate downstream system), not this Function App. Since 2026-09-02 this also catches anything the classifier would otherwise call `general_query` (see the note above) — an off-topic message gets exactly this same shape, no refusal.
```json
{ "type": "new", "belongto": "helpdesk", "status": "complete", "question": "How do I reset my VPN password?" }
```

This also covers questions **about** the travel-request/query-request features themselves — *"explain the travel request concept"*, *"how do I create a travel request"*, *"what's required to create a travel request"*, *"how to get a previous request"*. These are requests for an explanation, not an attempt to actually create or look something up, so they land here rather than starting the travel-creation or lookup flow. A real action — *"I want to travel to Cuba"*, *"show my tickets based on Cuba"* — still routes to `travel_request`/`query_request` as normal; only messages phrased as a question about how the feature works get redirected here.

**A bare greeting/address to the helpdesk with no issue stated** — *"helpdesk"*, *"hi helpdesk"*, *"hello helpdesk team"* — gets a canned welcome reply instead of being passed through as-is:
```json
{ "type": "new", "belongto": "helpdesk", "status": "complete", "question": "hi helpdesk", "message": "Hi! 👋 Welcome to the Helpdesk. I'm here to assist you.\n\nPlease let me know how I can help, and briefly explain the issue or trouble you're facing. I'll be happy to guide you through it." }
```
A message that also states a real issue (*"helpdesk, my VPN isn't connecting"*) is not treated as a greeting — it passes straight through as before, with no `message` field. This check is specific to `helpdesk` — a travel request is never affected by it.

### `query_request` — looking up something already submitted

Classifies which record type is being looked up (`travel_request` or `helpdesk_ticket`) and extracts search filters from the message. **This only classifies and extracts — it does not perform the actual lookup.** There is no database or storage layer in this codebase for either record type; whatever consumes this response is responsible for running the real search against the returned filters.

```json
{
  "type": "new", "belongto": "query_request", "status": "complete",
  "query": {
    "record_type": "travel_request",
    "filters": { "destinationCountry": "Cuba", "startDate": null, "endDate": null, "dateRaised": null, "ticketId": null, "description": null, "status": null },
    "sort": null,
    "limit": null
  }
}
```

If the record type isn't clear from the message (*"list all the requests"*, *"get my request"*, *"check the request"*), it asks first:
```json
{
  "type": "new", "belongto": "query_request", "status": "incomplete",
  "extracted": { "record_type": null, "filters": {}, "sort": null, "limit": null },
  "missing_field": "record_type",
  "follow_up_question": "Is that a travel request or a helpdesk ticket?"
}
```
Once the record type is known, it asks for at least one filter (or a sort/limit hint, see below) to narrow the search — unlike creating a travel request, not every field is required, just enough to search by. Answering **"all"**, **"everything"**, or **"any"** is recognized as a deliberate "no filter, show me everything" answer and completes the query immediately, rather than looping the same question forever.

Filter fields differ by record type, since the two don't share a schema:
- **`travel_request`:** `destinationCountry`, `startDate`, `endDate`, `dateRaised` (when it was submitted), `ticketId`, `description` (free-text keyword search), `status` — `null`, or an **array** of one or more of `"pending"`/`"approved"`/`"rejected"`/`"cancelled"` (an array, not a single string, so one message can ask for more than one status at once — see example below)
- **`helpdesk_ticket`:** `ticketId`, `dateRaised`, `subject` (free-text keyword search), `status` (a plain free-text ticket status like `"open"`/`"closed"` — unrelated to the fixed travel-status vocabulary above)

**`sort`/`limit` — since 2026-09-17.** Two fields alongside `filters`, at the same level as `record_type`, expressing ordering and a result count: `"sort"` is `"recent"` (most recent/latest/newest), `"oldest"` (earliest/first), or `null`. `"limit"` is the number of records asked for (`"latest 2"` → `2`), or `1` when `sort` is set but the phrasing implies a single record (`"my latest request"`), or `null` if neither is expressed. A sort/limit hint is enough on its own to complete the query even with zero filters — *"Can you tell me the status of my latest travel request?"* needs no destination, date, or status to be a complete, specific query:
```json
{
  "type": "new", "belongto": "query_request", "status": "complete",
  "query": {
    "record_type": "travel_request",
    "filters": { "destinationCountry": null, "startDate": null, "endDate": null, "dateRaised": null, "ticketId": null, "description": null, "status": null },
    "sort": "recent",
    "limit": 1
  }
}
```
A message naming more than one status resolves to all of them: *"show me the latest 2 approved and reject travel request ticket for cuba reason meeting client"* →
```json
{
  "type": "new", "belongto": "query_request", "status": "complete",
  "query": {
    "record_type": "travel_request",
    "filters": { "destinationCountry": "Cuba", "startDate": null, "endDate": null, "dateRaised": null, "ticketId": null, "description": "client meeting", "status": ["approved", "rejected"] },
    "sort": "recent",
    "limit": 2
  }
}
```
As with every filter in this codebase, **`sort`/`limit`/`status` are extracted hints only — no sorting, limiting, or status filtering actually happens here.** There's no persistence layer or lookup logic in this Function App; whatever consumes this response owns applying these values to the real search.

### `ambiguous_request` — wants to create something, doesn't say what

For messages like *"raise a request"* or *"request need to raise"* — a clear intent to create/raise/open something, but no topic at all. Asks which kind, then routes into the real `travel_request` or `helpdesk` flow using the original message once answered.
```json
{
  "type": "new", "belongto": "ambiguous_request", "status": "incomplete",
  "original_text": "raise a request",
  "missing_field": "request_type",
  "follow_up_question": "Is that a travel request or a helpdesk ticket?"
}
```

**The distinction that matters:** a *lookup* verb (get/show/see/check/list/find — "get **my** request", "list all **the** requests") always means `query_request`, never this category — regardless of whether "my"/"the"/"all" is used. Only a genuinely *indefinite, topic-less* create phrasing ("**a** request", "**a** ticket") with a create verb (raise/create/open/submit/file) lands here. If the message names a topic anywhere — "travel"/"trip", a destination, dates, or an IT term like "VPN"/"password" — it's classified directly into `travel_request` or `helpdesk` instead, with no extra question asked.

### `general_query` — internal-only, routed to `helpdesk`

Anything that doesn't fit the categories above, including a lookup or IT/workplace phrase that names a topic clearly outside travel or helpdesk (e.g. a movie ticket, a restaurant reservation) — those don't get force-fit into `query_request`/`helpdesk` just because they use a lookup verb or contain the word "ticket" (see `query_request` above for that scoping). The classifier still detects this case, but as of 2026-09-02 the response no longer surfaces it as its own outcome — it's routed to the `helpdesk` handler instead (deliberate direction, not a bug). See the `helpdesk` section above for the response shape; there is no separate `general_query` response anymore.

---

## Approved destination list

`destinationCountry` on a travel request isn't accepted freely — it's checked against a "CountryMaster" list maintained in SharePoint, one per deployment:

| Deployment | SharePoint site | Column holding the country name |
|---|---|---|
| Client | `ictsnd.sharepoint.com/sites/ICT/TravelRequest` | **Location Name** |
| Our side | `veeleaddev.sharepoint.com/sites/saunders/travelrequest` | **Country Name** |

A destination not on that list is rejected and re-asked — *"Sorry, {country} isn't a country we currently support for travel requests — could you confirm a different destination?"* — the same way an obviously malformed destination already was. The list is fetched once and cached for an hour (`COUNTRY_LIST_CACHE_TTL_SECONDS`), not re-fetched on every request. If the list genuinely can't be reached (SharePoint/Graph is down, or misconfigured), that's treated as a system error — *"Unable to verify the destination right now"* — never as "allow anything through," since the whole point of the list is to enforce which countries are permitted.

**Reading the list needs its own credential, and the two deployments need a genuinely different kind — not just different settings:**

- **Client side** uses the Function App's own **Managed Identity** — no secret stored anywhere. This works specifically because `saunders-travel-classify-func` and the client's SharePoint site live in the *same* Entra ID tenant.
- **Our side** uses `AZURE_TENANT_ID` / `SHAREPOINT_CLIENT_ID` / `SHAREPOINT_CLIENT_SECRET` — a traditional app-registration credential. This isn't a downgrade in security posture from a different choice; it's required, because `saunders-ai-agent-func` and `veeleaddev.sharepoint.com` sit in two *different* Entra ID tenants, and a Managed Identity is fundamentally bound to its own home tenant — it cannot be granted access to a resource in another one, regardless of what permissions get configured. Only an app registration (or workload identity federation) can be trusted across a tenant boundary that way.

The code auto-selects which of the two to use based on which settings are present — see `country_list.py`.

---

## API endpoints

| Route | Method | Purpose |
|---|---|---|
| `/api/classify` | POST | The main endpoint — everything above. Body: `{ "text": string, "previous_response"?: object }`. |
| `/api/health` | GET | Liveness check. No external calls. Returns `{ "status": "healthy" }`. |
| `/api/helpdesk-search` | POST | Isolated diagnostic endpoint — calls the third-party helpdesk search API directly with no classification, returning both its response and exactly what was sent to it. For verifying that integration in isolation, not part of the normal chatbot flow. |

All routes require a function key by default (`http_auth_level=FUNCTION`), except where the hosting environment is configured otherwise.

---

## Azure infrastructure

This code is deployed to **two** separate Azure environments — the client's own subscription (the one the frontend team's chatbot actually talks to) and VeeLead's own subscription (used for our own testing/verification). They are entirely separate resources with separate settings; a deploy to one does not affect the other, and both need to be redeployed to when this code changes.

### Client side — ICT SND subscription

Everything below lives in one resource group so it's easy to find and, if ever needed, tear down as a unit.

| Resource | Name | Purpose |
|---|---|---|
| Resource group | `saunders-ai-agent` | Everything below lives here. Region: **Australia East**. |
| Function App | `saunders-travel-classify-func` | Hosts this codebase. Python 3.11, Linux, **Consumption plan** — scales to zero and only costs money per execution, no charge while idle. Live at `https://saunders-travel-classify-func.azurewebsites.net`. |
| Azure OpenAI | `Saundersaimodels` | Used for classification and extraction (see [Configuration](#configuration)). Same region as the Function App, so there's no cross-region network hop on the AI call. |
| Storage account | `saundersaistorage` | Required by the Function App runtime itself (deployment packages, trigger bookkeeping). Holds no business/request data — nothing about an actual travel request or helpdesk ticket is stored here. |
| Application Insights | `saunders-travel-classify-func` | Request logging and monitoring for the Function App, auto-created alongside it. |

**History:** this Function App was originally deployed as `saunders-helpdesk-func`. Azure doesn't support renaming a Function App in place (the name is part of its URL), so getting to the current name meant standing up a new app (`saunders-travel-classify-func`) with the same settings, cutting over, and then deleting the old one. If you have the old URL saved anywhere, it no longer exists — use the one above.

### Our side — VeeLead's own subscription ("Azure Learning")

| Resource | Name | Purpose |
|---|---|---|
| Resource group | `rg-saunders-ai-agent` | Everything below lives here. |
| Function App | `saunders-ai-agent-func` | Same codebase, deployed separately for our own testing/verification. Python 3.11, Linux, Consumption plan. Live at `https://saunders-ai-agent-func.azurewebsites.net`. |

**Access control (both environments):** every route requires a function key (`?code=...` on the URL) — there is no anonymous access on either deployment. CORS is currently open (`*`) on the client side so any web frontend can call the API from the browser; this should be narrowed to the actual chatbot's domain before it's relied on for real traffic rather than testing.

---

## Project structure

| File | Purpose |
|---|---|
| `function_app.py` | Entry point. HTTP trigger, input validation, the classification call, and routing between the five category handlers. |
| `travel_handler.py` | Everything for `travel_request` — extraction, field-by-field validation, the stateless follow-up loop. |
| `query_handler.py` | Everything for `query_request` — record-type classification, filter extraction, the "show all" shortcut. |
| `country_list.py` | Fetches and caches the approved-destination list from SharePoint — see [Approved destination list](#approved-destination-list). |
| `helpdesk_api_client.py` | Thin client for the third-party helpdesk search API, used only by the `/api/helpdesk-search` diagnostic endpoint. |
| `helpdesk_search.py` | Superseded — an earlier SharePoint-based *helpdesk* search implementation, kept in the repo but not imported or used anywhere anymore. Not to be confused with `country_list.py`, which is a different, current SharePoint integration for a different purpose. |
| `host.json` | Azure Functions host configuration (logging, extension bundle, timeout). |
| `requirements.txt` | Python dependencies: `azure-functions`, `requests`, `azure-identity`. |
| `local.settings.json.example` | Template for local configuration — copy to `local.settings.json` and fill in real values (never committed; see below). |

---

## Configuration

Set via Function App settings in Azure (or `local.settings.json` for local development — copy `local.settings.json.example`, never commit the real file). No secret values are recorded here or in that example file.

| Setting | Purpose |
|---|---|
| `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_API_KEY` | Azure OpenAI resource used for classification and extraction. |
| `AZURE_OPENAI_API_VERSION` | Defaults to `2024-08-01-preview`. |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Defaults to `gpt-4o-mini` in code. Currently configured to `gpt-5` in the client environment (switched from `gpt-4o` on 2026-09-17, since `gpt-4o` is marked "Deprecating" on the client's Azure OpenAI resource). `gpt-5` is a reasoning-family model — already handled automatically by `ai_client.py`'s model-family detection (no code changes needed to switch), but responses are noticeably slower per call than a classic chat model. |
| `MAX_INPUT_CHARS` | Max length of an incoming message. Defaults to `4000`. |
| `REQUEST_TIMEOUT_SECONDS` | Per-call timeout to Azure OpenAI / the helpdesk API. Defaults to `20`. |
| `MAX_RETRIES` / `RETRY_BASE_DELAY_SECONDS` | Retry/backoff for transient failures. Defaults to `3` / `1.5`. |
| `LOW_CONFIDENCE_THRESHOLD` | Confidence below which a classification is logged as low-confidence (still routed, not blocked). Defaults to `0.6`. |
| `HELPDESK_API_BASE` / `HELPDESK_API_KEY` | Third-party helpdesk search API, used only by the `/api/helpdesk-search` diagnostic endpoint. |
| `SHAREPOINT_HOSTNAME` / `SHAREPOINT_SITE_PATH` / `SHAREPOINT_COUNTRY_LIST_ID` / `SHAREPOINT_COUNTRY_COLUMN_DISPLAY_NAME` | Where the approved-country list lives — see [Approved destination list](#approved-destination-list) for the real values on each deployment. |
| `COUNTRY_LIST_CACHE_TTL_SECONDS` | How long the fetched country list is cached before refetching. Defaults to `3600` (1 hour). |
| `AZURE_TENANT_ID` / `SHAREPOINT_CLIENT_ID` / `SHAREPOINT_CLIENT_SECRET` | Only needed on **our side** — the app-registration credential used to read the country list cross-tenant (see above). Not set on the client side, which uses its Managed Identity instead. |

---

## Known limitations

- **No persistence layer.** Travel requests and helpdesk tickets are classified/extracted but never stored anywhere by this codebase. `query_request` returns filters to search by, not actual results — the real search/database lookup is owned by whatever system consumes this API.
- **No cross-topic conversation memory.** State only persists within a single in-progress request (a travel request or query being filled in). `helpdesk` doesn't remember anything from earlier turns.
- **Off-topic messages are now indistinguishable from real helpdesk questions.** Since 2026-09-02, `general_query` (still detected internally) is routed to the `helpdesk` handler with no refusal — whatever consumes `helpdesk` responses downstream will now also receive things like movie-ticket requests or "tell me a joke", framed identically to a real IT question. No history/memory logic exists for either — that would need a persistence layer and a stable per-user identifier, neither of which exist yet.
- **The two environments use different chat models.** The client's Function App is configured with `AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5`; VeeLead's own is still configured with `gpt-4o-mini`. `gpt-4o-mini` is measurably less reliable at following the classifier's more detailed rules on edge-case phrasing (confirmed 2026-08-31: "raise a travel request" — rule 3's own example — consistently misclassified as `ambiguous_request` on the `gpt-4o-mini` deployment while classifying correctly every time on `gpt-4o`). If VeeLead's environment needs classification parity with the client's, switch its `AZURE_OPENAI_CHAT_DEPLOYMENT` setting (cost/latency tradeoff to weigh first).

---

## Changelog

Dated, most recent first. Reflects changes to the deployed behavior of this Function App — not internal testing activity.

- **2026-09-17** — `query_request` (looking up an existing travel request/ticket) now extracts sorting/limiting/status hints, not just exact-match filters. Two new top-level fields alongside `filters`: `"sort"` (`"recent"`/`"oldest"`/`null`) and `"limit"` (an integer count, or `1` when a sort is implied for a single record, or `null`). `travel_request`'s filters also gained `"status"` — `null`, or an **array** of `"pending"`/`"approved"`/`"rejected"`/`"cancelled"` (an array so a single message can name more than one, e.g. "approved and rejected" -> `["approved", "rejected"]`). A sort/limit hint alone (no other filter at all) is now enough to complete the query — *"Can you tell me the status of my latest travel request?"* was previously stuck asking for a filter it didn't need. No lookup/sorting logic was added here — these are still just extracted hints for whatever system performs the real search. Live and verified on client (23/23 existing regression checks unaffected, 8/8 new scenario checks passing).
- **2026-09-17** — Switched the client's chat model from `gpt-4o` to `gpt-5`, since `gpt-4o` is now marked "Deprecating" on the client's Azure OpenAI resource. `gpt-5` is a reasoning-family model, already supported automatically by the existing model-family-aware request builder — no code changes were needed, just the `AZURE_OPENAI_CHAT_DEPLOYMENT` app setting. Verified with a full side-by-side comparison against `gpt-4o` on the classification rulebook and field extraction (identical categories and identical extracted fields) before switching, then the full 23-scenario regression suite live on the client, 23/23 passing after the switch. Responses are slower per call than `gpt-4o` (reasoning models take longer to respond) — worth watching if latency becomes noticeable.
- **2026-09-16** — When a trip's end date is before its start date, both dates are now cleared and re-collected from scratch — starting with the start date — instead of assuming only the end date was wrong and silently keeping the (possibly also mistaken) start date. *"The dates you gave don't work together — the end date is before the start date. Please check and enter the start date again, then the end date. What's the start date for the trip to Japan?"* There's no way to know from a swapped/conflicting pair which side was actually mistyped, and patching just one risked exactly the kind of stuck loop a downstream system consuming this response could get into. Live and verified on client.
- **2026-09-16** — A start date in the past now gets a clear, explicit explanation — *"Sorry, we can't raise a travel request for a date in the past — you can only raise a request for today or a future date. Could you confirm the correct start date..."* — instead of the previous neutral "...is in the past — could you confirm the correct start date?", which didn't actually say why it wasn't allowed. Also taught extraction to resolve RELATIVE date phrases ("last week", "previous week to next week", "past two weeks") into real calendar dates using today's date as the reference point, rather than leaving them unparsed — so a vague past-referring phrase now correctly reaches the same clear rejection message instead of silently having no date at all. A genuine future date (relative or absolute) is unaffected. Live and verified on client.
- **2026-09-15** — Two follow-ups to a "Can I..."/"Could I..." classifier fix made the same day. First, a real regression it introduced: "Can I raise a **travel** request for another person?" — which explicitly names "travel" — was being swept into the same `ambiguous_request` treatment as the topic-less case, instead of going straight to `travel_request`. The rule now explicitly reasons "delete 'Can I', then classify what's left" — a named topic always wins over the no-topic carve-out, "Can I" or not. Second, a related extraction bug: once the disambiguation question was answered, an on-behalf phrase combined with a short reply (e.g. "Can I raise a request for another person?" + "travel") could get dropped entirely, returning a request with no fields captured at all instead of asking who it's for. Fixed by explicitly telling extraction that a question phrasing is never a reason to treat trip detail as absent. Live and verified on client (dev testing paused per current direction).
- **2026-09-15** — A message phrased as a polite question — "Can I create a request for my friend instead of myself?" — was getting swept into the classifier's "explanation question" rule and routed straight to `helpdesk`, instead of asking "Is that a travel request or a helpdesk ticket?" like the same message without "Can I" would. The rule now treats a "Can I"/"Could I" opener as cosmetic — classify what's left as if it weren't there — unless what's left is itself a genuine question about how/whether a feature works (which still correctly stays `helpdesk`). Live and verified on client.

- **2026-09-11** — `startDate`/`endDate` no longer ask for a time at all — in practice users don't give one when asked, so the extra question just added friction. A date alone is now sufficient to complete the field; a volunteered time is still captured if given, just never required or asked for. While removing the time question, also found and fixed a real bug it exposed: a bare time answering the date question (e.g. "9:30am" with no date anywhere) could get silently combined with *today's* date instead of being rejected — fixed so a time with nothing to attach it to is never accepted as a date and never defaults to any invented date; the question is asked again instead. Live and verified on both environments.
- **2026-09-10** — Two follow-ups to yesterday's `requestFor` (raising a request for someone else) feature. First, a real bug: a valid email address answering the "who is this for" question could get silently ignored and the same question re-asked, if the AI extraction step happened to fail to parse a bare email with no surrounding words — fixed with a deterministic check (an "@" in the reply is unambiguous, no AI interpretation needed) that also fixed an inline email in the very first message getting mistakenly shortened to a guessed name (e.g. "john@company.com" -> "John"). Second, a policy change: a name alone ("John", "my manager") is no longer treated as sufficient — an actual email address is now always required to consider `requestFor` resolved, with a name-only answer followed by one more targeted question ("Could you share John's email address?") instead of skipping ahead. Live and verified on both environments.

- **2026-09-10** — Travel requests now support being raised on someone else's behalf. New field `requestFor` (null by default — the ordinary self-request case is completely unaffected) captures phrasing like "for my manager", "on behalf of my friend", "for John" wherever it's mentioned in the conversation, not just as the answer to a dedicated question. Checked FIRST, before reason — if what's captured is a bare placeholder with no name or email ("someone", "my manager"), exactly one question asks *"Who is this travel request for? Could you share their name or email address?"*; a real name or email given anywhere (including inline in the first message) skips the question entirely. Live and verified on both environments — 23/23 existing regression checks unaffected, 17/17 new scenario checks passing.
- **2026-09-09** — The SharePoint approved-country list itself was expanded by its owner from 56 narrow entries to a full 195-country list — not a code change, but worth noting: Germany, Japan, France, and most other everyday destinations are now legitimately approved where they were rejected before. Confirmed the AI country-match fallback (added 2026-09-08) adapts automatically to renamed entries on the new list (e.g. "Burma" -> "Myanmar", "North Korea" -> "Korea (North)") with no code changes needed.
- **2026-09-09** — Same misclassification pattern as "show suspended country" (below), a different trigger: a lookup verb next to an IT/hardware-flavored topic with no actual submitted ticket behind it — e.g. "can you get warranty details for my laptop" — was inconsistently routed to `query_request` instead of `helpdesk` (~4/5 of the time). The lookup rule now requires the thing being looked up to actually be a submitted request/ticket, not just an IT/hardware-adjacent noun — a plain hardware question now reaches the existing IT-topic rule and goes to `helpdesk` every time. Live and verified on both environments (10/10 consistent).
- **2026-09-08** — General reference/policy questions with no personal record to look up — e.g. "show suspended country", "show approved country list" — were inconsistently classified: usually `query_request` (asking "Is that a travel request or a helpdesk ticket?", which doesn't make sense here since there's no actual record), occasionally `helpdesk` on the exact same input, a genuine model-level judgment-call inconsistency rather than a code bug. The classifier now explicitly excludes this kind of question from the lookup rule that leads to `query_request` — it's not a lookup of something the user submitted, just a reference question — so it routes to `helpdesk` every time instead. A real personal-record lookup ("show my travel request to Cuba", "check my flight status") is unaffected. Live and verified on both environments (10/10 consistent, was ~9/10 before).
- **2026-09-08** — Fixed a bug where an unusually-formatted approved-list entry (e.g. "Gaza / Palestinian Territories") could get rejected even when the user typed it back exactly — extraction's own "normalize to standard English name" step was rewriting it to "Palestinian Territories" or "Palestine" before the approval check ran, and neither rewritten form matched the list's exact text. The approval check now falls back to an AI match against the full approved list when an exact match fails, and corrects the stored value to the list's exact text on a match (e.g. "Burma" -> "Burma (Myanmar)", "Trinidad" -> "Trinidad and Tobago") — a genuinely different country (e.g. "South Korea") is still correctly rejected. Live and verified on both environments.
- **2026-09-08** — A destination given up front (e.g. "i want to go to japan") that isn't on the approved list is now rejected on that very first message, instead of asking for the reason first and only rejecting the destination afterward. Reason is still asked first whenever no destination has been given yet — this only changes the case where an already-stated destination is invalid. Live and verified on both environments.

- **2026-09-03** — The travel-request follow-up order is now a fixed **reason → destination → start date/time → end date/time**, always — reverted from a brief same-day change that asked reason last. Live and verified on both environments.
- **2026-09-03** — `startTime`/`endTime` are removed; `startDate`/`endDate` now each carry both the date and the time in one value (`"YYYY-MM-DDTHH:mm"`, naive wall-clock, e.g. `"2027-08-05T09:30"`), matching a UI change to one combined calendar+clock control per leg boundary instead of two separate ones. A date whose time isn't known yet is a legitimate `"YYYY-MM-DD"` value — `missing_field_path` stays on the same field (`0.startDate`/`0.endDate`) and the follow-up asks specifically for the time (*"What time on 5 Aug 2027 does the trip to Cuba start?"*) rather than repeating the whole question. Ambiguous-time-returns-null behavior is unchanged, just applied to this merged field — the date-only value is left untouched rather than guessed at. Live and verified on both environments.
- **2026-09-03** — A message that just addresses/greets the helpdesk with no issue stated yet — "helpdesk", "hi helpdesk", "hello helpdesk team" — now gets a canned welcome reply asking the user to describe their issue, instead of being passed straight through as if "helpdesk" itself were the question. A message that also states a real issue (e.g. "helpdesk, my VPN isn't connecting") is unaffected — travel requests are also unaffected. Live and verified on both environments.
- **2026-09-02** — Travel requests now capture **departure/return time** per leg (`startTime`/`endTime`), alongside the existing dates. Always `"HH:mm"` 24-hour on the wire regardless of how the user phrased it ("9:30am", "half nine", "6 in the evening" all normalize the same way dates already do); a genuinely ambiguous time ("at 7") comes back `null` and is asked again rather than guessed. Asked in departure-then-return pairs — **start date → start time → end date → end time** — rather than both dates first, then both times. Live and verified on both environments.
- **2026-09-02** — A full-sentence answer to a destination/reason follow-up question (e.g. "i want to travel to cuba" answering "could you confirm a different destination?") was getting stored as the destination verbatim — the whole sentence, not just "Cuba" — instead of extracting the actual value from it. Now properly extracts just the value the same way fresh extraction does, correcting spelling along the way. Live and verified on both environments.
- **2026-09-02** — `general_query` no longer appears as an outcome — the classifier still detects it internally (unchanged), but it's now routed to the `helpdesk` handler instead of returning its polite refusal. An off-topic message (e.g. "book ticket for movie") now gets `"belongto": "helpdesk"` with no refusal text, identical in shape to a real helpdesk question. Deliberate, explicit direction — not a bug fix. Live on both environments.
- **2026-09-02** — A typo'd trip reason (e.g. "cleint vist" for "client visit") could be wrongly rejected as not a legitimate business reason — the business-reason check didn't explicitly tolerate typos the way every other classification check in this system does. Added that instruction; verified on both environments.
- **2026-09-02** — Added cancel-intent detection for a pending travel request — "cancel", "exit", "start over", "discard", and similar now end the request outright (`status: "cancelled"`) instead of being mistaken for an unrelated reply or, worse, literally stored as an answer to whatever field was being asked. Live and verified on both environments.
- **2026-09-02** — A lookup that explicitly names "travel request" or "helpdesk ticket" (e.g. "first we can see travel request") could still be asked "Is that a travel request or a helpdesk ticket?" if filler wording diluted the signal — it now reliably recognizes the named type regardless of surrounding phrasing. Live and verified on both environments.
- **2026-09-02** — "I want to travel" (bare travel intent, no stated purpose) no longer gets the word "travel" itself extracted as the reason — it now correctly asks "What's the reason for your trip?" instead of skipping straight to destination. A message that does state a purpose (e.g. "I want to travel for business") still extracts it directly as before. Live and verified on both environments.
- **2026-09-01** — The business-reason check now explicitly recognizes personal leisure/entertainment (watching a movie, a concert, a sporting event, shopping, sightseeing) as non-business, not just vacations/honeymoons/family visits — it previously only listed a few specific personal-reason examples and didn't reliably generalize to other clearly-personal activities. Live and verified on both environments.
- **2026-09-01** — A one-word trip reason (e.g. "travel", "meeting", "trip", "business", "training") is now accepted instead of being wrongly rejected as not a legitimate business reason — the check only needs to rule out something clearly personal or meaningless, not demand detail the later questions will collect anyway. Meaningless single words (a greeting, an article, an unrelated noun) are still correctly rejected. Live and verified on both environments.
- **2026-09-01** — A message asking WHICH category a request falls under (e.g. "I'd like to submit a request but I'm not sure of the correct category — could you advise?") now correctly routes to `ambiguous_request` and asks travel-or-helpdesk, instead of being misread as a how-does-this-feature-work question and routed to `helpdesk`. Live and verified on both environments.
- **2026-09-01** — Resolving "Is that a travel request or a helpdesk ticket?" (ambiguous_request) or "which record type?" (query_request) now recovers any real detail stated in that same reply (e.g. "travel request, I'm going to Cuba" recovers Cuba as the destination) instead of discarding it — previously only the original message was re-examined, so anything added in the disambiguation reply itself was silently lost and had to be repeated. Live and verified on both environments.
- **2026-09-01** — Fixed a bug where a travel request that stated the reason before naming a destination (e.g. "I want to travel for meeting client for business purpose") had its reason silently discarded and was re-asked for it, instead of being asked for the missing destination. The extraction prompt previously required a named destination to create a leg entry at all — it now creates one from any stated detail (destination, dates, or reason), so nothing given up front gets thrown away. Live and verified on both environments.
- **2026-08-31** — Meta/how-to questions about raising a request are now reliably routed to `helpdesk` even when phrased with no topic at all (e.g. "how and where can I raise a request") or combined with stated travel intent (e.g. "I want to travel to Cuba, how do I do that?") — previously these could fall through to `ambiguous_request` or `travel_request` instead of being recognized as a process question. Live and verified on both environments.
- **2026-08-31** — A travel request's reason is now checked for being a legitimate business reason (client visit, conference, training, etc.) — a clearly personal reason (a marriage proposal, a vacation, visiting family) is rejected and re-asked, same pattern as the destination-country check. Live and verified on both environments.
- **2026-08-31** — Fixed a misclassification where a lookup for something clearly unrelated to travel/IT (e.g. "get my movie ticket to bahubali 2") was being routed to `query_request` just because it used a lookup verb ("get") next to the word "ticket" — it was then force-fit into a travel-request or helpdesk-ticket search, returning the user's unrelated travel history. Lookups and ticket-mentions are now only routed to `query_request`/`helpdesk` when the request/ticket is unqualified (e.g. "my request") or plausibly travel/IT-specific; anything else now falls to `general_query`, which returns a polite refusal instead of passing the raw question through. Live and verified on both environments.
- **2026-08-31** — Answering "Is that a travel request or a helpdesk ticket?" (or the equivalent record-type lookup question) with a typo (e.g. "helpdisk") now resolves correctly instead of silently re-asking the exact same question — both disambiguation checks were a hardcoded keyword match with no typo tolerance, unlike the rest of the system; both now use a small AI call instead. Live and verified on both environments.
- **2026-08-31** — A pending travel request no longer silently absorbs an unrelated reply (e.g. a helpdesk complaint typed mid-flow) into the field it was waiting on. It's now checked against the pending question first; an unrelated reply gets a "you have a travel request in progress — finish it or refresh the chat" notice instead. Live and verified on both environments.
- **2026-08-31** — An unsupported destination country is now flagged immediately when given, before asking for the reason or dates — previously the reason was asked first even when the destination was already known to be rejected. Live and verified on both environments.
- **2026-08-31** — A misspelled destination country (e.g. "phillipines") is now corrected to its standard name ("Philippines") before being checked against the approved-country list, instead of being rejected as unsupported due to the typo. Live and verified on both environments.
- **2026-08-31** — `destinationCountry` on a travel request is now checked against an approved-country list maintained in SharePoint (see [Approved destination list](#approved-destination-list)) — a well-formed but unapproved country is now rejected and re-asked instead of accepted. Live and verified on both environments.
- **2026-08-29** — Questions asking about the travel-request/helpdesk-ticket features themselves ("explain the travel request concept," "how to create a travel request," "what's required to create a travel request," "how to get a previous request") now route to `helpdesk` instead of being misrouted into starting the actual creation/lookup flow. Deployed to both environments.
- **2026-08-28** — Deployed the current codebase to both the client-side Function App (`saunders-travel-classify-func`) and VeeLead's own Function App (`saunders-ai-agent-func`) for the first time in this project's history — everything in this changelog up to this point was only pushed to GitHub before today, not actually live on either environment. Both verified healthy afterward.
- **2026-08-28** — A multi-destination travel message where every leg is otherwise complete now asks for the reason once for all of them ("What's the reason for the trips?") instead of once per leg — one answer covers every trip in the message.
- **2026-08-27** — Cut over to the current Function App name, `saunders-travel-classify-func` (previously `saunders-helpdesk-func` — Azure doesn't support renaming in place, so this was a new app with the same settings, then the old one was deleted). If you have the old URL saved, it no longer exists.
- **2026-08-27** — Migrated the deployment from the development subscription to the client's own Azure subscription (ICT SND, Australia East), including the client's own Azure OpenAI resource.
- **2026-08-26** — `query_request` now recognizes "all"/"everything"/"any" as a deliberate answer to "which one would you like to narrow down by," completing the lookup with no filter instead of re-asking the same question forever.
- **2026-08-26** — Broadened lookup detection in the classifier: phrasings like "list all **the** requests" or "check the request" (not just "**my** request") are now correctly recognized as a lookup (`query_request`), not a create request.
- **2026-08-26** — Fixed inconsistent routing between "get my request" (lookup) and "raise a request" (create) — both are now reliably distinguished by the classifier.
- **2026-08-21** — Added `ambiguous_request`: messages like "raise a request" that want to create something but name no topic now get asked which kind before proceeding, instead of being misrouted.
- **2026-08-19** — Added `query_request` for looking up existing travel requests/helpdesk tickets.
- **2026-08-19** — Reordered travel request follow-up questions: reason is now asked before destination and dates.
- **2026-08-19** — Fixed a date-handling bug where a follow-up answer with no year could resolve to the wrong year.
- **2026-08-18** — Added a guard against the AI extraction occasionally folding a date into the destination field; extended the smart follow-up fallback to every field, not just dates; fixed ambiguous numeric dates (e.g. "7-9-2026") being read month-first instead of day-first.
- **2026-08-17** — Replaced the SharePoint-based helpdesk search with a third-party helpdesk API integration; built the travel request intake flow (extraction, validation, stateless follow-up); added the `general_query` category and flattened the response shape.
- **2026-08-17** — Initial version: classify + route between `helpdesk` and `travel_request`.
