#!/usr/bin/env python
"""Stage 0 mock or stage 3 live five-essay pilot; both conditions by default."""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.dont_write_bytecode = True

import yaml
from para_agent.artifacts import create_run, file_hash, write_json
from para_agent.config import load_config, project_path
from para_agent.data.loader import load_mock_essays
from para_agent.data.pilot import load_stage3_pilot
from para_agent.diagnose.mock import MockDiagnoser
from para_agent.kb.lookup import KnowledgeBase
from para_agent.llm.client import Client, MockTransport
from para_agent.run import incremental, whole_once
from para_agent.run.common import Services
from para_agent.eval.policy import load_evaluation_policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Stage 0: no network or model calls")
    mode.add_argument("--stage", type=int, choices=[3], help="Live execution is available only for stage 3")
    parser.add_argument("--split-manifest", type=Path, help="Audited stage 2 pilot manifest; required for live runs")
    parser.add_argument("--condition", choices=["both", "incremental", "whole_once"], default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    policy_path = project_path(cfg.paths.evaluation_policy)
    policy = load_evaluation_policy(policy_path)
    if cfg.diag_unit == "hybrid":
        parser.error("Selective hybrid is deferred; use diag_unit=both to retain both diagnoses")
    if args.mock and (cfg.llm.agent.provider != "mock" or cfg.llm.judge.provider != "mock"):
        parser.error("Stage 0 requires mock agent and judge providers")
    if not args.mock and (cfg.llm.agent.provider not in {"openai", "anthropic"} or args.split_manifest is None):
        parser.error("Stage 3 requires an OpenAI/Anthropic agent config and --split-manifest; judge is not called")
    contracts_path = project_path(cfg.paths.contracts)
    contracts = yaml.safe_load(contracts_path.read_text(encoding="utf-8"))
    kb_path, key_map = project_path(cfg.kb.features), project_path(cfg.kb.feature_key_map)
    kb = KnowledgeBase(kb_path, key_map)
    rubric_map_path = project_path(cfg.paths.rubric_diag_map)
    rubric_map = yaml.safe_load(rubric_map_path.read_text(encoding="utf-8"))
    if set(rubric_map) != set(contracts["rubric_keys"]):
        parser.error("rubric_diag_map must use the exact known Kanana rubric keys")
    if set(kb.key_map.values()) != set(contracts["feature_keys"]):
        parser.error("Explicit KB key mapping does not match the feature contract")
    if args.mock:
        fixture = project_path(cfg.data.mock_input)
        essays = load_mock_essays(fixture, min_paras=cfg.data.min_paras,
                                  max_paras=cfg.data.max_paras, limit=cfg.data.n_essays)
        input_info = {"input_file": str(fixture), "input_sha256": file_hash(fixture)}
    else:
        essays, input_info = load_stage3_pilot(args.split_manifest, cfg)
    prompts = project_path(cfg.paths.prompts)
    responses = PROJECT_ROOT / "tests/fixtures/mock_responses.json"
    conditions = ["whole_once", "incremental"] if args.condition == "both" else [args.condition]
    output = args.output_dir or project_path(cfg.paths.runs)
    cache = args.cache_dir or project_path(cfg.paths.cache)
    run = create_run(output, PROJECT_ROOT, cfg.model_dump(), {
        "stage": 0 if args.mock else 3, "mock": args.mock,
        "evaluation_policy_version": policy['version'],
        "evaluation_policy_sha256": file_hash(policy_path),
        "data_kind": "synthetic_fixture" if args.mock else "heldout_test_pilot", "conditions": conditions,
        "essay_count": len(essays), **input_info,
        "config_source": str(args.config.resolve()), "config_sha256": file_hash(args.config),
        "output_root": str(output.resolve()), "cache_dir": str(cache.resolve()),
        "prompt_hashes": {path.name: file_hash(path) for path in sorted(prompts.glob("*.md"))},
        "kb_sha256": file_hash(kb_path), "feature_key_map_sha256": file_hash(key_map),
        "contracts_sha256": file_hash(contracts_path), "rubric_diag_map_sha256": file_hash(rubric_map_path),
        "rubric_unit_map_sha256": file_hash(project_path(cfg.paths.rubric_unit_map)),
        "mock_responses_sha256": file_hash(responses) if args.mock else None,
        "implementation_hashes": {str(path.relative_to(PROJECT_ROOT)): file_hash(path)
                                  for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))},
    })
    write_json(run / 'evaluation_policy.json', policy)
    expected_calls = sum(2 * len(essays) if c == "whole_once" else
                         3 * sum(len(e.paragraphs) for e in essays) for c in conditions)
    report = {"run_dir": str(run.resolve()), "ok": False, "stage": 0 if args.mock else 3,
              "mock": args.mock, "conditions": conditions, "essays": len(essays),
              "paragraphs": [len(e.paragraphs) for e in essays], "expected_logical_calls": expected_calls}
    transport = diagnoser = None
    started = time.perf_counter()
    if not args.mock:
        print(f"run_dir: {run}", file=sys.stderr, flush=True)
    try:
        if args.mock:
            transport = MockTransport(responses)
            diagnoser = MockDiagnoser(contracts["rubric_keys"], contracts["feature_keys"])
        else:
            from para_agent.diagnose.service import build_diagnoser
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            if cfg.llm.agent.provider == "openai":
                from para_agent.llm.openai_transport import OpenAITransport
                transport_type = OpenAITransport
            else:
                from para_agent.llm.anthropic_transport import AnthropicTransport
                transport_type = AnthropicTransport
            transport = transport_type(cfg.llm.agent, project_path(cfg.llm.env_file), max_calls=expected_calls * 3)
            diagnoser = build_diagnoser(cfg, log_path=run / "diagnostic_calls.jsonl")
        client = Client(model=cfg.llm.agent, transport=transport, prompts=prompts,
                        cache=cache, call_log=run / "llm_calls.jsonl")
        services = Services(diagnoser, client, kb, rubric_map, run)
        runners = {"whole_once": whole_once.run, "incremental": incremental.run}
        for condition in conditions:
            for essay in essays:
                runners[condition](essay, cfg, services)
                if not args.mock:
                    print(f"completed: {condition} {essay.essay_id} ({len(essay.paragraphs)} paragraphs)",
                          file=sys.stderr, flush=True)
        report["ok"] = True
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        # Provider exception strings can echo credentials. Only record the type/status.
        report["http_status"] = getattr(exc, "status_code", None)
        raise RuntimeError(f"Run failed ({type(exc).__name__}); inspect {run}") from None
    except KeyboardInterrupt:
        report.update(error_type="KeyboardInterrupt", interrupted=True)
        raise SystemExit(130) from None
    finally:
        rows = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text().splitlines()]
        turns = [json.loads(p.read_text()) for p in run.glob('*/*/turn_*.json')]
        report.update(latency_s=time.perf_counter() - started,
                      failed_anonymization_turns=sum(t.get('turn_status')=='failed_anonymization' for t in turns),
                      rejected_error_notes=sum(t.get('memory_update_validation',{}).get('rejected_count',0) for t in turns),
                      mock_transport_calls=transport.call_count if transport and args.mock else 0,
                      real_api_calls=transport.call_count if transport and not args.mock else 0,
                      cache_hits=sum(row["cache_hit"] for row in rows),
                      parse_repairs=sum(row["attempt"] > 0 for row in rows),
                      call_statuses=dict(Counter(row.get("status", "interrupted") for row in rows)),
                      tokens={key: sum(row["tokens"].get(key, 0) for row in rows)
                              for key in ("input_tokens", "output_tokens", "total_tokens")},
                      turn_logs=len(list(run.glob("*/*/turn_*.json"))),
                      final_texts=len(list(run.glob("*/*/final.json"))))
        write_json(run / "acceptance.json", report)
        try:
            if diagnoser is not None and hasattr(diagnoser, "close"):
                diagnoser.close()
        finally:
            if transport is not None and hasattr(transport, "close"):
                transport.close()
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

