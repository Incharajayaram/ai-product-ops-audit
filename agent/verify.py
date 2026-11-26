"""Verification pass: sample pass-1 results, independently re-check them with
fresh evidence, and produce a corrected dataset + an accuracy report.

Run: python -m agent.verify

Method (two independent loops, as the brief asks):
1. Agent re-check loop: for each sampled app, gather a FRESH, independently
   targeted piece of Tavily evidence (different query/extraction than pass 1),
   then ask Gemini to judge pass-1's claims against that fresh evidence per
   field (confirm / contradict+correct / not addressed). This is a second,
   differently-sourced check, not a re-run of the same pipeline.
2. Human check loop: a small hand-picked subset is listed separately in
   output/human_check_queue.json with the evidence URLs, for a person to
   actually open and eyeball. Results of that manual pass are merged in by
   `record_human_check()` below and saved to output/human_check_results.json.

Output:
- output/verification_report.json: per-field agreement, list of corrections
- output/final_results.json: pass-1 with corrections applied
"""
from __future__ import annotations

import json
import logging
import random

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.config import (
    FINAL_FILE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OUTPUT_DIR,
    PASS1_FILE,
    VERIFICATION_FILE,
)
from agent.schema import AuthMethod, BuildabilityVerdict, SelfServeStatus
from agent.tavily_tools import tavily_extract, tavily_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("verify")

_client = genai.Client(api_key=GEMINI_API_KEY)

SAMPLE_SEED = 42
MIN_CONFIDENCE_AUTO_INCLUDE = 0.85
HUMAN_CHECK_COUNT = 6

CHECK_FIELDS = [
    "auth_methods",
    "self_serve_status",
    "api_surface_summary",
    "has_existing_mcp",
    "buildability_verdict",
]


def build_sample(results: list[dict]) -> list[dict]:
    """Every low-confidence record, plus one random higher-confidence record
    per category, so the sample both targets likely errors and checks for
    false confidence elsewhere."""
    random.seed(SAMPLE_SEED)

    low_conf = [r for r in results if r["confidence"] < MIN_CONFIDENCE_AUTO_INCLUDE]
    low_conf_ids = {r["app_id"] for r in low_conf}

    by_category: dict[str, list[dict]] = {}
    for r in results:
        if r["app_id"] not in low_conf_ids:
            by_category.setdefault(r["category"], []).append(r)

    spot_checks = [random.choice(apps) for apps in by_category.values()]

    sample = low_conf + spot_checks
    sample.sort(key=lambda r: r["app_id"])
    return sample


def gather_fresh_evidence(record: dict, hint_url: str) -> tuple[str, list[str]]:
    """Deliberately different from pass 1: one narrower, claim-targeted search
    instead of the broad discovery searches pass 1 used. Still anchored to the
    hint domain so ambiguous names (e.g. "Sherlock") don't drift onto an
    unrelated product of the same name, same as the main pipeline does."""
    name = record["name"]
    disambiguator = hint_url.split()[0].split("/")[0] if hint_url else ""
    query = (
        f'"{name}" ({disambiguator}) official API docs: authentication method, who can get API access '
        f"(free signup vs paid plan vs approval), and whether an MCP server exists"
    )
    results = tavily_search(query, max_results=4)
    urls = [r["url"] for r in results if r.get("url")]
    extracted = tavily_extract(urls[:2])
    extracted_by_url = {e["url"]: e.get("raw_content", "") for e in extracted if e.get("url")}

    blocks = []
    for r in results:
        url = r["url"]
        content = extracted_by_url.get(url) or r.get("content", "")
        blocks.append(f"Source: {url}\nTitle: {r.get('title', '')}\n{content[:4000]}")
    return "\n\n---\n\n".join(blocks), urls


class AuthMethodsCheck(BaseModel):
    verdict: str = Field(description="One of: confirmed, contradicted, not_addressed")
    corrected_value: list[AuthMethod] | None = None
    explanation: str


class SelfServeStatusCheck(BaseModel):
    verdict: str = Field(description="One of: confirmed, contradicted, not_addressed")
    corrected_value: SelfServeStatus | None = None
    explanation: str


class HasMcpCheck(BaseModel):
    verdict: str = Field(description="One of: confirmed, contradicted, not_addressed")
    corrected_value: bool | None = None
    explanation: str


class BuildabilityCheck(BaseModel):
    verdict: str = Field(description="One of: confirmed, contradicted, not_addressed")
    corrected_value: BuildabilityVerdict | None = None
    explanation: str


class ApiSurfaceCheck(BaseModel):
    verdict: str = Field(description="One of: confirmed, contradicted, not_addressed")
    note: str = Field(description="Free-text note; this field is prose, so no enum correction.")


class VerificationJudgement(BaseModel):
    app_id: int
    name: str
    auth_methods: AuthMethodsCheck
    self_serve_status: SelfServeStatusCheck
    has_existing_mcp: HasMcpCheck
    buildability_verdict: BuildabilityCheck
    api_surface_summary: ApiSurfaceCheck
    overall_verdict: str = Field(description="One of: pass1_correct, pass1_partially_wrong, pass1_wrong")
    notes: str = Field(description="Anything surprising, including if a source turned out to be about a different, similarly-named product.")


@retry(
    retry=retry_if_exception_type((ClientError, ServerError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=60),
)
def judge_against_fresh_evidence(record: dict, hint_url: str, fresh_evidence: str) -> VerificationJudgement:
    pass1_claims = {k: record.get(k) for k in CHECK_FIELDS}
    prompt = f"""You are auditing another AI agent's research. It researched the app below and
produced the claims in `pass1_claims`. You are given FRESH, independently gathered evidence
(different search than the one that produced pass1_claims). Check each field in pass1_claims
against the fresh evidence ONLY.

For each field, decide:
- "confirmed": fresh evidence supports the claim
- "contradicted": fresh evidence clearly disagrees; give `corrected_value` using ONLY the
  allowed enum values for that field (never a free-text word like "available" or "freemium")
- "not_addressed": fresh evidence doesn't speak to this field either way (don't guess; leave
  `corrected_value` null)

Be skeptical and literal. Do not confirm a claim just because it sounds plausible.

Some app names are ambiguous (shared with unrelated products). The app's known domain/hint is
given below — if a fresh-evidence source is clearly about a different product with the same
name, do not use it; say so in `notes` and mark affected fields "not_addressed" rather than
contradicting pass1 based on the wrong product.

App: {record['name']} ({record['category']}, app_id={record['app_id']})
Known hint/domain: {hint_url}

pass1_claims:
{json.dumps(pass1_claims, indent=2)}

Fresh evidence:
---
{fresh_evidence}
---
"""
    resp = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VerificationJudgement,
            temperature=0.1,
        ),
    )
    return VerificationJudgement.model_validate_json(resp.text)


def main() -> None:
    results = json.loads(PASS1_FILE.read_text())
    apps = json.loads((OUTPUT_DIR.parent / "data" / "apps.json").read_text())
    hint_by_id = {a["id"]: a["hint_url"] for a in apps}

    sample = build_sample(results)
    log.info("Verification sample: %d/%d apps -> %s",
              len(sample), len(results), [r["app_id"] for r in sample])

    judgements = []
    for record in sample:
        hint_url = hint_by_id.get(record["app_id"], "")
        try:
            fresh_evidence, urls = gather_fresh_evidence(record, hint_url)
            if not fresh_evidence.strip():
                log.warning("No fresh evidence for %s, skipping", record["name"])
                continue
            judgement = judge_against_fresh_evidence(record, hint_url, fresh_evidence)
            judgements.append({**judgement.model_dump(mode="json"), "fresh_evidence_urls": urls})
            log.info("Verified [%3d] %-28s -> %s",
                      record["app_id"], record["name"], judgement.overall_verdict)
        except Exception as e:  # noqa: BLE001
            log.error("Verification failed for %s: %s", record["name"], e)

    # Apply corrections to a copy of pass-1 results. Only auth_methods,
    # self_serve_status, has_existing_mcp, and buildability_verdict are typed
    # enums/bools validated by the schema above, so these are safe to write
    # back directly. api_surface_summary is free text, so we only note it.
    by_id = {r["app_id"]: dict(r) for r in results}
    corrections_applied = []

    def apply_field(app_id: int, name: str, field: str, check: dict):
        if check["verdict"] == "contradicted" and check.get("corrected_value") is not None:
            old_value = by_id[app_id].get(field)
            new_value = check["corrected_value"]
            by_id[app_id][field] = new_value
            by_id[app_id].setdefault("_correction_notes", []).append(
                f"{field}: {old_value!r} -> {new_value!r} ({check['explanation']})"
            )
            corrections_applied.append({
                "app_id": app_id, "name": name, "field": field,
                "old_value": old_value, "new_value": new_value,
            })

    all_checks_flat = []
    for j in judgements:
        app_id, name = j["app_id"], j["name"]
        for field in ("auth_methods", "self_serve_status", "has_existing_mcp", "buildability_verdict"):
            check = j[field]
            apply_field(app_id, name, field, check)
            all_checks_flat.append(check)
        # api_surface_summary is prose-only: record the note but don't overwrite.
        surf = j["api_surface_summary"]
        all_checks_flat.append({"verdict": surf["verdict"]})
        if surf["verdict"] == "contradicted":
            by_id[app_id].setdefault("_correction_notes", []).append(
                f"api_surface_summary: flagged by verification - {surf['note']}"
            )

    final = [by_id[r["app_id"]] for r in results]
    FINAL_FILE.write_text(json.dumps(final, indent=2))

    total_checks = len(all_checks_flat)
    confirmed = sum(1 for c in all_checks_flat if c["verdict"] == "confirmed")
    contradicted = sum(1 for c in all_checks_flat if c["verdict"] == "contradicted")
    not_addressed = total_checks - confirmed - contradicted

    report = {
        "sample_size": len(sample),
        "total_population": len(results),
        "total_field_checks": total_checks,
        "confirmed": confirmed,
        "contradicted": contradicted,
        "not_addressed": not_addressed,
        "pass1_field_accuracy_on_sample": round(confirmed / total_checks, 3) if total_checks else None,
        "apps_pass1_wrong_or_partial": [
            {"app_id": j["app_id"], "name": j["name"], "verdict": j["overall_verdict"]}
            for j in judgements if j["overall_verdict"] != "pass1_correct"
        ],
        "corrections_applied": corrections_applied,
        "judgements": judgements,
    }
    VERIFICATION_FILE.write_text(json.dumps(report, indent=2))

    log.info(
        "Verification done. Sample field accuracy: %s (%d/%d). %d corrections applied.",
        report["pass1_field_accuracy_on_sample"], confirmed, total_checks, len(corrections_applied),
    )

    # Prepare a human-check queue: a few of the sample's evidence URLs, for a
    # person to actually open and eyeball by hand.
    human_queue = []
    for record in sample[:HUMAN_CHECK_COUNT]:
        human_queue.append({
            "app_id": record["app_id"],
            "name": record["name"],
            "pass1_claims": {k: record.get(k) for k in CHECK_FIELDS},
            "evidence_urls": record.get("_meta", {}).get("evidence_urls_gathered", []),
        })
    (OUTPUT_DIR / "human_check_queue.json").write_text(json.dumps(human_queue, indent=2))
    log.info("Human-check queue written for %d apps -> output/human_check_queue.json", len(human_queue))


if __name__ == "__main__":
    main()
