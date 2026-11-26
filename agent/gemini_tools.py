"""Structured extraction via Gemini."""
from __future__ import annotations

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.config import GEMINI_API_KEY, GEMINI_MODEL
from agent.schema import AppResearch

_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a precise product-ops researcher. You assess whether a SaaS \
app's public API could power an AI agent toolkit today.

Rules:
- Base every answer ONLY on the evidence snippets given to you. Never use outside knowledge \
of the app beyond what's in the evidence.
- If the evidence does not clearly answer a field, say so: use "Unknown"/"unknown" enum \
values, lower your confidence, and explain the gap in `notes`. Do not guess or assume a \
common default (e.g. do not assume OAuth2 just because most SaaS apps use it).
- `self_serve_status` is about whether a DEVELOPER can get API credentials themselves for \
free or on a trial, vs needing a paid plan, admin approval, or a sales/partnership conversation.
- `evidence_urls` must be a subset of the URLs actually present in the evidence block.
- `confidence` should be LOW (under 0.5) if the evidence is thin, generic, or contradictory.
- Some app names are ambiguous (shared with unrelated products). Before using a source, check
  it actually matches the given `hint` (domain/description). If a source is clearly about a
  different product with the same name, ignore it, note the mix-up in `notes`, and lower
  `confidence` if this leaves you with little usable evidence.
"""


@retry(
    retry=retry_if_exception_type((ClientError, ServerError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=60),
)
def extract_app_research(app: dict, evidence_block: str) -> AppResearch:
    prompt = f"""{SYSTEM_PROMPT}

App to research:
app_id: {app['id']}
name: {app['name']}
category: {app['category']}
hint: {app['hint_url']}

Evidence gathered from web search and page extraction:
---
{evidence_block}
---

Fill in the structured assessment for this app based only on the evidence above."""

    resp = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=AppResearch,
            temperature=0.1,
        ),
    )
    return AppResearch.model_validate_json(resp.text)
