"""Thin wrapper around the Composio session for calling Tavily tools."""
from __future__ import annotations

import logging

from composio import Composio
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.config import COMPOSIO_API_KEY, COMPOSIO_USER_ID

log = logging.getLogger(__name__)

TAVILY_AUTH_CONFIG_SEARCH_NAME = "Tavily (research agent)"

_composio: Composio | None = None
_session = None


def _get_session():
    global _composio, _session
    if _session is not None:
        return _session

    _composio = Composio(api_key=COMPOSIO_API_KEY)
    configs = _composio.auth_configs.list(search=TAVILY_AUTH_CONFIG_SEARCH_NAME)
    if not configs.items:
        raise RuntimeError(
            "No Tavily auth config found. Run `python -m agent.composio_setup` first."
        )
    auth_config_id = configs.items[0].id

    _session = _composio.create(
        user_id=COMPOSIO_USER_ID,
        toolkits=["tavily"],
        auth_configs={"tavily": auth_config_id},
    )
    return _session


class TransientToolError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(TransientToolError),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
)
def _execute(tool_slug: str, arguments: dict) -> dict:
    session = _get_session()
    result = session.execute(tool_slug, arguments=arguments)
    data = result.model_dump() if hasattr(result, "model_dump") else result

    if data.get("error"):
        err = str(data["error"])
        if any(code in err for code in ("429", "rate limit", "timeout", "5")):
            raise TransientToolError(err)
        raise RuntimeError(f"{tool_slug} failed: {err}")

    return data


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """Returns a list of {title, url, content, score}."""
    data = _execute("TAVILY_SEARCH", {"query": query, "max_results": max_results})
    return data.get("data", {}).get("results", []) or []


def tavily_extract(urls: list[str]) -> list[dict]:
    """Returns a list of {url, raw_content}."""
    if not urls:
        return []
    data = _execute("TAVILY_EXTRACT", {"urls": urls, "format": "markdown"})
    payload = data.get("data", {})
    return payload.get("results", []) or []
