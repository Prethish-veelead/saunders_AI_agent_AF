"""
Shared Azure OpenAI chat-completion request builder.
------------------------------------------------------

Every AI call in this codebase goes through here so a per-environment
model swap never has to touch each of the ~9 call sites individually —
this project already runs different deployments per environment (the
client's own vs. VeeLead's), and that gap is only getting wider as
newer model families show up.

The one thing that genuinely differs by model family:

  - Classic chat models (gpt-4o, gpt-4o-mini, and earlier) accept a
    custom `temperature` (this codebase always pins it to 0.0, for
    reproducible classification) and cap the response with `max_tokens`.
  - Newer "reasoning" models (the gpt-5 family, o1, o3, and similar)
    reject any `temperature` other than their own default outright —
    confirmed live: "Unsupported value: 'temperature' does not support
    0.0 with this model." They also use `max_completion_tokens` instead
    of `max_tokens` — confirmed live: "'max_tokens' is not supported
    with this model. Use 'max_completion_tokens' instead." — and spend
    part of that budget on invisible "reasoning tokens" before writing
    the visible answer: at the model's default reasoning effort, or
    even "minimal" with too tight a budget, the ENTIRE budget can be
    consumed by reasoning with zero visible output (finish_reason
    "length", empty content — reproduced live at a 20-token budget,
    this codebase's smallest). Setting reasoning_effort to "minimal"
    reliably brings reasoning_tokens to 0 for this codebase's short
    classification/extraction prompts, provided the budget has enough
    headroom — empirically ~50 tokens minimum; MIN_REASONING_TOKENS
    below uses a larger margin for safety.
"""

_REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")

# Empirically ~50 tokens was the reliable floor for this codebase's
# smallest checks (a one-field JSON response); doubled for margin.
MIN_REASONING_MODEL_TOKENS = 100


def is_reasoning_model(deployment_name: str) -> bool:
    name = (deployment_name or "").strip().lower()
    return name.startswith(_REASONING_MODEL_PREFIXES)


def build_chat_payload(
    deployment_name: str,
    messages: list,
    max_tokens: int,
    response_format: dict = None,
) -> dict:
    """Builds a chat-completion payload appropriate for whichever model
    family `deployment_name` names. `max_tokens` is the caller's normal
    (classic-model) budget — raised automatically for a reasoning model
    if it's below the empirically-safe floor, so callers don't each
    need to know about that quirk.
    """
    payload: dict = {"messages": messages}
    if response_format is not None:
        payload["response_format"] = response_format

    if is_reasoning_model(deployment_name):
        payload["max_completion_tokens"] = max(max_tokens, MIN_REASONING_MODEL_TOKENS)
        payload["reasoning_effort"] = "minimal"
        # temperature intentionally omitted — these models reject any
        # value other than their own default.
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0.0

    return payload
