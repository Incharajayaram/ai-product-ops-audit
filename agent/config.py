"""Environment and shared constants for the research agent."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")

COMPOSIO_API_KEY = os.environ["COMPOSIO_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

# A stable user_id for Composio sessions. This agent researches public docs,
# not a real end-user's accounts, so one fixed id for the whole pipeline is fine.
COMPOSIO_USER_ID = "research_agent"

GEMINI_MODEL = "gemini-flash-lite-latest"

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

APPS_FILE = DATA_DIR / "apps.json"
PASS1_FILE = OUTPUT_DIR / "pass1_results.json"
VERIFICATION_FILE = OUTPUT_DIR / "verification_report.json"
FINAL_FILE = OUTPUT_DIR / "final_results.json"
