"""Structured output schema for one researched app."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AuthMethod(str, Enum):
    OAUTH2 = "OAuth2"
    API_KEY = "API key"
    BASIC = "Basic"
    TOKEN = "Token"
    NO_AUTH = "No auth"
    OTHER = "Other"
    UNKNOWN = "Unknown"


class SelfServeStatus(str, Enum):
    SELF_SERVE_FREE = "self_serve_free"
    SELF_SERVE_TRIAL = "self_serve_trial"
    GATED_PAID_PLAN = "gated_paid_plan"
    GATED_APPROVAL = "gated_approval"
    GATED_PARTNERSHIP = "gated_partnership"
    UNKNOWN = "unknown"


class BuildabilityVerdict(str, Enum):
    READY_TODAY = "ready_today"
    POSSIBLE_WITH_WORKAROUND = "possible_with_workaround"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


class AppResearch(BaseModel):
    app_id: int
    name: str
    category: str

    one_line_description: str = Field(description="What the app does, in one sentence.")

    auth_methods: list[AuthMethod] = Field(
        description="All authentication methods the app's API documentation supports."
    )

    self_serve_status: SelfServeStatus
    gating_detail: str = Field(
        description="One sentence on how a developer actually gets credentials: "
        "free self-serve signup, trial, paid plan required, needs admin/sales approval, etc."
    )

    api_surface_summary: str = Field(
        description="One or two sentences on the API surface: REST/GraphQL, roughly how many "
        "endpoints/resources or how broad, and whether it's well documented."
    )
    has_existing_mcp: Optional[bool] = Field(
        default=None, description="Whether an official or well-known community MCP server exists for this app."
    )
    mcp_detail: Optional[str] = None

    buildability_verdict: BuildabilityVerdict
    main_blocker: Optional[str] = Field(
        default=None, description="The single biggest blocker to building an agent toolkit today, if any."
    )

    evidence_urls: list[str] = Field(
        description="The specific doc/article URLs that support the answers above."
    )

    confidence: float = Field(ge=0, le=1, description="Model's own confidence in these findings, 0-1.")
    notes: Optional[str] = Field(
        default=None, description="Anything surprising, conflicting, or that needed a judgment call."
    )
