# Agent-Ready Audit

Composio's AI Product Ops Intern take-home: research 100 apps' auth/API/MCP buildability with
an agent, cluster the patterns, and verify the results.

**Live case study:** see the deployment link shared with the submission.

## What's here

```
agent/
  config.py          env + constants
  composio_setup.py  one-time: wires a Tavily API key into Composio as a connected account
  tavily_tools.py    search/extract via the Composio session (Tavily toolkit)
  gemini_tools.py    structured extraction (Gemini + Pydantic schema)
  schema.py          the AppResearch record schema
  research.py        pass-1 pipeline: researches all 100 apps, concurrent + resumable
  verify.py          verification pass: independent re-check + corrections
data/
  apps.json          the 100-app list from the assignment brief
output/
  pass1_results.json         raw pass-1 results (100 apps)
  verification_report.json   sampling + accuracy report
  human_check_results.json   3 manually hand-checked apps
  final_results.json         pass-1 with verification corrections applied
site/
  index.html, data.json      the deployed case-study page
```

## How it works

1. **Research (pass 1).** For each app, two Tavily searches (general + commercial-access) through
   a Composio session, full-page extraction of the top-ranked official-looking URLs, then Gemini
   turns that evidence into a typed record (auth methods, self-serve status, API surface, MCP
   status, buildability verdict, evidence URLs, confidence). Runs 4 apps concurrently, flushes
   each result to `output/pass1_results.jsonl` immediately so a crash loses at most one in-flight
   app.
2. **Verify.** A stratified sample (every low-confidence app + one random app per category, 23
   total) gets a second, independently-worded search, and Gemini judges pass 1's claims against
   that fresh evidence only — confirm, contradict-and-correct (using the same validated enum,
   never free text), or "evidence doesn't say." A separate human pass hand-checked 3 apps by
   actually reading the cited docs.
3. **Report.** `output/final_results.json` + `output/verification_report.json` feed directly into
   `site/data.json`, which the static page renders.

Known issues found and fixed along the way (see the deployed page's "The agent" section for
details): a name-collision bug on ambiguous app names, a Gemini free-tier daily-quota wall that
killed the first full run at app #8, and a verification-schema bug that produced unusable
free-text corrections before being rewritten as typed enums.

## Running it yourself

Requires Python 3.10+, and three free API keys (no credit card for any of them):

- **Composio** — [dashboard.composio.dev](https://dashboard.composio.dev), Platform project →
  Getting Started → API Key.
- **Tavily** — [app.tavily.com](https://app.tavily.com), free tier (1,000 credits/month).
- **Gemini** — [aistudio.google.com/apikey](https://aistudio.google.com/apikey), free tier.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cat > .env.local <<EOF
COMPOSIO_API_KEY=...
TAVILY_API_KEY=...
GEMINI_API_KEY=...
EOF

python -m agent.composio_setup   # one-time: connects Tavily to Composio
python -m agent.research         # researches all 100 apps -> output/pass1_results.json
python -m agent.verify           # verification pass -> output/final_results.json
```

`site/index.html` reads `site/data.json`; after a research/verify run, regenerate it by copying
the relevant fields from `output/final_results.json` + `output/verification_report.json` +
`output/human_check_results.json` (see the consolidation step documented in the page's own
methodology section).

## Stack

[Composio](https://composio.dev) Python SDK (sessions) · [Tavily](https://tavily.com) search/extract ·
[Google Gemini](https://ai.google.dev) (`gemini-flash-lite-latest`) for structured extraction ·
Pydantic for schema validation · a static HTML/JS page for the deliverable, no framework.
