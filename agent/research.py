"""Core research pipeline: for each app, search + extract via Tavily (through
Composio), then structure the findings with Gemini.

Run: python -m agent.research
Writes incrementally to output/pass1_results.jsonl (one JSON object per line),
then consolidates into output/pass1_results.json at the end.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from agent.config import APPS_FILE, OUTPUT_DIR, PASS1_FILE
from agent.gemini_tools import extract_app_research
from agent.schema import AppResearch
from agent.tavily_tools import tavily_extract, tavily_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("research")

JSONL_FILE = OUTPUT_DIR / "pass1_results.jsonl"
MAX_WORKERS = 4
EVIDENCE_CHAR_LIMIT = 6000  # per extracted page, to keep the Gemini prompt bounded


def _looks_official(url: str, app_name: str, hint_url: str) -> int:
    """Heuristic score: higher = more likely to be the app's own official docs."""
    host = urlparse(url).netloc.lower()
    name_tokens = app_name.lower().replace(".", " ").split()
    score = 0
    if any(tok in host for tok in name_tokens if len(tok) > 2):
        score += 2
    if any(seg in host for seg in ("docs.", "developer.", "developers.", "api.")):
        score += 2
    hint_domain = hint_url.split()[0].split("/")[0].lower()
    if hint_domain and hint_domain in host:
        score += 3
    return score


def gather_evidence(app: dict) -> tuple[str, list[str]]:
    name, hint = app["name"], app["hint_url"]
    # Disambiguate common/generic names (e.g. "Sherlock", "Twenty", "Clay") by
    # anchoring the query to the known domain/hint from the assignment's app list.
    disambiguator = hint.split()[0].split("/")[0] if hint else ""

    results_a = tavily_search(
        f'"{name}" ({disambiguator}) API documentation authentication', max_results=5
    )
    results_b = tavily_search(
        f'"{name}" ({disambiguator}) developer API key sign up pricing OR partner approval OR MCP server',
        max_results=5,
    )

    by_url: dict[str, dict] = {}
    for r in (results_a + results_b):
        if r.get("url"):
            by_url.setdefault(r["url"], r)

    ranked = sorted(
        by_url.values(),
        key=lambda r: (_looks_official(r["url"], name, hint), r.get("score", 0)),
        reverse=True,
    )

    top_urls_to_extract = [r["url"] for r in ranked[:2]]
    extracted = tavily_extract(top_urls_to_extract)
    extracted_by_url = {e["url"]: e.get("raw_content", "") for e in extracted if e.get("url")}

    blocks = []
    evidence_urls = []
    for r in ranked[:6]:
        url = r["url"]
        evidence_urls.append(url)
        if url in extracted_by_url and extracted_by_url[url]:
            content = extracted_by_url[url][:EVIDENCE_CHAR_LIMIT]
            blocks.append(f"Source (full page extract): {url}\nTitle: {r.get('title', '')}\n{content}")
        else:
            content = (r.get("content") or "")[:2000]
            blocks.append(f"Source (search snippet): {url}\nTitle: {r.get('title', '')}\n{content}")

    return "\n\n---\n\n".join(blocks), evidence_urls


def research_app(app: dict) -> dict:
    t0 = time.time()
    try:
        evidence_block, evidence_urls = gather_evidence(app)
        if not evidence_block.strip():
            raise RuntimeError("no search evidence found")

        result: AppResearch = extract_app_research(app, evidence_block)
        record = result.model_dump(mode="json")
        record["_meta"] = {
            "status": "ok",
            "seconds": round(time.time() - t0, 1),
            "evidence_urls_gathered": evidence_urls,
        }
        log.info("OK   [%3d] %-28s verdict=%s self_serve=%s",
                  app["id"], app["name"], result.buildability_verdict, result.self_serve_status)
        return record

    except Exception as e:  # noqa: BLE001 - we want to keep going across 100 apps
        log.error("FAIL [%3d] %-28s %s", app["id"], app["name"], e)
        return {
            "app_id": app["id"],
            "name": app["name"],
            "category": app["category"],
            "_meta": {
                "status": "error",
                "error": str(e),
                "seconds": round(time.time() - t0, 1),
            },
        }


def main() -> None:
    apps = json.loads(APPS_FILE.read_text())

    # Resume support: skip apps already recorded in the JSONL file.
    done_ids: set[int] = set()
    if JSONL_FILE.exists():
        for line in JSONL_FILE.read_text().splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["app_id"])
    todo = [a for a in apps if a["id"] not in done_ids]
    log.info("Resuming: %d/%d apps already done, %d remaining", len(done_ids), len(apps), len(todo))

    with JSONL_FILE.open("a") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(research_app, app): app for app in todo}
        for fut in as_completed(futures):
            record = fut.result()
            f.write(json.dumps(record) + "\n")
            f.flush()

    all_records = [json.loads(line) for line in JSONL_FILE.read_text().splitlines() if line.strip()]
    all_records.sort(key=lambda r: r["app_id"])
    PASS1_FILE.write_text(json.dumps(all_records, indent=2))

    ok = sum(1 for r in all_records if r.get("_meta", {}).get("status") == "ok")
    log.info("Done. %d/%d succeeded. Consolidated -> %s", ok, len(all_records), PASS1_FILE)


if __name__ == "__main__":
    main()
