#!/usr/bin/env python
"""Stage 1 acceptance check using unchanged local models and a synthetic essay."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv

from para_agent.artifacts import create_run, file_hash, write_json
from para_agent.config import load_config, project_path
from para_agent.diagnose.contracts import read_legacy_contract
from para_agent.diagnose.service import build_diagnoser
from para_agent.kb.validate import validate_kb


def source_hashes(cfg):
    paths = set((Path(cfg.features.package_path) / "essay_scoring_llm").rglob("*.py"))
    paths.update((Path(cfg.scorer.repository_path) / "verak/v3/score").rglob("*.py"))
    paths.update(Path(path) for path in read_legacy_contract(cfg.features.package_path)["sources"])
    paths.update([Path(cfg.scorer.source_config), project_path(cfg.kb.features)])
    return {str(path): file_hash(path) for path in sorted(paths)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "tests/fixtures/diagnosis_essay.json")
    args = parser.parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cfg = load_config(args.config)
    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    before = source_hashes(cfg)
    kb_result = validate_kb(project_path(cfg.kb.features), project_path(cfg.kb.feature_key_map),
                            cfg.features.package_path)
    require(kb_result["ok"], f"KB validation failed: {kb_result['errors']}")
    run = create_run(project_path(cfg.paths.runs), PROJECT_ROOT, cfg.model_dump(),
                     {"stage": 1, "input": str(args.input.resolve()), "input_sha256": file_hash(args.input),
                      "synthetic": True, "llm_api_calls": 0, "legacy_sources": before})
    write_json(run / "input.json", fixture)
    write_json(run / "kb_validation.json", kb_result)
    diagnoser = build_diagnoser(cfg, log_path=run / "diagnostic_calls.jsonl")
    print(f"run_dir: {run}", flush=True)
    started = time.perf_counter()
    report = {"stage": 1, "ok": False, "synthetic": True, "llm_api_calls": 0, "cases": {}}
    try:
        import torch
        report["cuda"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                          "scorer_gpu": diagnoser.scorer.settings["scorer"]["gpu"],
                          "feature_gpu": cfg.features.gpu}
        require(report["cuda"]["available"], "The existing Kanana scorer requires CUDA")
        question, paragraphs = fixture["prompt"], fixture["paragraphs"]
        require(len(paragraphs) >= 3, "The fixture must have at least three existing paragraphs")
        cases = {"para": (paragraphs[1], paragraphs[:1], "para"),
                 "cum_single": (paragraphs[0], [], "cum"),
                 "cum": (paragraphs[1], paragraphs[:1], "cum"),
                 "whole": ("\n\n".join(paragraphs), [], "cum")}
        for name, (text, prefix, unit) in cases.items():
            outputs = [diagnoser.diagnose(text, prefix, unit, question, use_cache=False)
                       for _ in range(2)]
            equal = outputs[0] == outputs[1]
            write_json(run / name / "diagnosis.json", {"first": outputs[0], "repeat": outputs[1],
                                                       "same_without_cache": equal})
            report["cases"][name] = {"same_without_cache": equal,
                                     "rubrics": len(outputs[0].rubric_scores),
                                     "features": len(outputs[0].features),
                                     "null_features": [k for k, v in outputs[0].features.items() if v is None]}
            require(equal, f"Repeated uncached {name} diagnosis differs; see saved outputs")
            require(len(outputs[0].rubric_scores) == 8 and len(outputs[0].features) == 29,
                    f"Invalid {name} output key count")
            if name in {"para", "cum_single"}:
                require(all(outputs[0].features[key] is None for key, scope in diagnoser.features.scopes.items()
                            if scope == "cum"), "Cumulative-only values must be None for a single paragraph")
            print(f"{name}: 8 rubrics, 29 features, uncached repeat identical", flush=True)

        pair = diagnoser.diagnose(paragraphs[1], paragraphs[:1], "both", question)
        write_json(run / "both" / "diagnosis.json", pair)
        for name in ("para", "cum"):
            saved = json.loads((run / name / "diagnosis.json").read_text(encoding="utf-8"))["first"]
            require(getattr(pair, name).model_dump() == saved, f"both.{name} differs from the standalone diagnosis")
        report["both_keeps_separate_values"] = True

        diagnoser.scorer.score("\n\n".join(paragraphs), question, use_cache=True)
        last = json.loads((run / "diagnostic_calls.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        require(last["result"]["cache_hit"], "Expected a native scorer cache hit")
        report["native_cache_hit"] = True

        from verak.v3.score.kanana import InputTooLong
        try:
            diagnoser.scorer.score("문장입니다. " * 4096, question, use_cache=False)
        except InputTooLong as exc:
            report["overflow"] = {"input_tokens": exc.tokens, "limit": exc.limit,
                                  "truncated": False, "rejected": True}
        else:
            raise AssertionError("Expected overflow rejection without truncation")
        report["legacy_files_unchanged"] = source_hashes(cfg) == before
        require(report["legacy_files_unchanged"], "A legacy source or KB file changed during diagnosis")
        report["ok"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["latency_s"] = time.perf_counter() - started
        write_json(run / "acceptance.json", report)
        diagnoser.close()
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
