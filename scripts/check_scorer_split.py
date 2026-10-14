#!/usr/bin/env python
"""Check local Kanana connectivity for the prepared split; no revisions or LLM APIs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.dont_write_bytecode = True

from para_agent.artifacts import create_run, file_hash, write_json
from para_agent.config import load_config, project_path
from para_agent.data.loader import load_essays
from para_agent.diagnose.contracts import read_legacy_contract
from para_agent.diagnose.scorer_adapter import KananaScorerAdapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--split-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split = Path(manifest["split"])
    if manifest.get("stage") != 2 or manifest.get("ok") is not True or file_hash(split) != manifest["split_sha256"]:
        raise ValueError("Expected an unchanged, successful stage 2 split")
    essays = load_essays(split, min_paras=cfg.data.min_paras, max_paras=cfg.data.max_paras)
    keys = read_legacy_contract(cfg.scorer.package_path)["rubric_keys"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    run = create_run(project_path(cfg.paths.runs), PROJECT_ROOT, cfg.model_dump(),
                     {"stage": 2, "validation": "scorer_metadata", "llm_api_calls": 0,
                      "split_manifest": str(args.split_manifest.resolve()), "split_sha256": file_hash(split)})
    scorer = KananaScorerAdapter(cfg.scorer, keys, log_path=run / "diagnostic_calls.jsonl")
    started = time.perf_counter()
    report = {"ok": False, "stage": 2, "scorer_gpu": scorer.settings["scorer"]["gpu"],
              "genre_metadata": scorer.metadata_provenance, "essay_count": len(essays),
              "llm_api_calls": 0, "revision_calls": 0, "results": []}
    print(f"run_dir: {run}", flush=True)
    try:
        for essay in essays:
            scores, rationales = scorer.score("\n\n".join(essay.paragraphs), essay.prompt, use_cache=False)
            if list(scores) != keys:
                raise AssertionError("Scorer rubric keys or order changed")
            native = json.loads((run / "diagnostic_calls.jsonl").read_text(encoding="utf-8").splitlines()[-1])["result"]
            record = {"essay_id": essay.essay_id, "rubric_scores": scores, "rubric_rationales": rationales,
                      "genre": native["genre"], "input_tokens": native["input_tokens"], "cache_hit": native["cache_hit"]}
            write_json(run / essay.essay_id / "score.json", record)
            report["results"].append(record)
            print(f"{essay.essay_id}: 8 scores, {native['genre']}, {native['input_tokens']} tokens", flush=True)
        report["ok"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["latency_s"] = time.perf_counter() - started
        write_json(run / "acceptance.json", report)
        scorer.close()
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
