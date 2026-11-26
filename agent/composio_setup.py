"""One-time setup: connect Composio's TAVILY toolkit using our own free Tavily API key.

Run directly: python -m agent.composio_setup

Idempotent: safe to re-run. It reuses an existing auth config / connected
account for TAVILY + COMPOSIO_USER_ID instead of creating duplicates.
"""
from __future__ import annotations

from composio import Composio
from composio.types import auth_scheme

from agent.config import COMPOSIO_API_KEY, COMPOSIO_USER_ID, TAVILY_API_KEY

TOOLKIT = "tavily"


def get_or_create_auth_config(composio: Composio) -> str:
    existing = composio.auth_configs.list(search="Tavily (research agent)")
    for cfg in existing.items:
        return cfg.id

    auth_config = composio.auth_configs.create(
        toolkit=TOOLKIT,
        options={
            "type": "use_custom_auth",
            "auth_scheme": "API_KEY",
            "name": "Tavily (research agent)",
            "credentials": {},
        },
    )
    return auth_config.id


def get_or_create_connected_account(composio: Composio, auth_config_id: str) -> str:
    existing = composio.connected_accounts.list(
        user_ids=[COMPOSIO_USER_ID],
        auth_config_ids=[auth_config_id],
    )
    for acc in existing.items:
        if acc.status == "ACTIVE":
            return acc.id

    connection = composio.connected_accounts.initiate(
        user_id=COMPOSIO_USER_ID,
        auth_config_id=auth_config_id,
        config=auth_scheme.api_key({"api_key": TAVILY_API_KEY}),
    )
    return connection.id


def main() -> None:
    composio = Composio(api_key=COMPOSIO_API_KEY)
    auth_config_id = get_or_create_auth_config(composio)
    print(f"auth_config_id = {auth_config_id}")

    connected_account_id = get_or_create_connected_account(composio, auth_config_id)
    print(f"connected_account_id = {connected_account_id}")

    # Smoke test: open a session scoped to just Tavily and list its tools.
    session = composio.create(user_id=COMPOSIO_USER_ID, toolkits=[TOOLKIT])
    tools = session.tools()
    tool_names = sorted({t["function"]["name"] for t in tools if "function" in t})
    print(f"Session ready. {len(tools)} tools visible, including: "
          f"{[n for n in tool_names if 'SEARCH' in n or 'EXTRACT' in n]}")


if __name__ == "__main__":
    main()
