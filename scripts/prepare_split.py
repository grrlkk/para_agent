#!/usr/bin/env python
"""Stage 2: create a reproducible held-out pilot from original paragraph boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from para_agent.artifacts import create_run, file_hash, json_text, write_json
from para_agent.config import load_config, project_path
from para_agent.data.loader import load_essays, load_heldout_candidates, stratified_sample
from para_agent.diagnose.contracts import read_legacy_contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--n-essays", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.n_essays is not None:
        cfg.data.n_essays = args.n_essays
    if args.seed is not None:
        cfg.data.seed = args.seed
    if cfg.data.source is None:
        parser.error("data.source must identify the audited test split and its original records")
    contract = read_legacy_contract(cfg.features.package_path)
    candidates, audit, excluded = load_heldout_candidates(cfg.data.source, contract["rubric_keys"],
                                                        min_paras=cfg.data.min_paras, max_paras=cfg.data.max_paras)
    selected, sampling = stratified_sample(candidates, count=cfg.data.n_essays, seed=cfg.data.seed)
    # Sampling must reproduce from the same pool regardless of input iteration order.
    repeated, _ = stratified_sample(list(reversed(candidates)), count=cfg.data.n_essays, seed=cfg.data.seed)
    if selected != repeated:
        raise AssertionError("Seeded selection is not reproducible")
    run = create_run(project_path(cfg.paths.runs), PROJECT_ROOT, cfg.model_dump(),
                     {"stage": 2, "mock": False, "llm_api_calls": 0, "model_inference_calls": 0,
                      "config_source": str(args.config.resolve()), "config_sha256": file_hash(args.config),
                      "contract_sources": contract["sources"]})
    output = project_path(cfg.data.split_output)
    output.mkdir(parents=True, exist_ok=True)
    split = output / f"pilot_{len(selected)}_{run.name}.jsonl"
    with split.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json_text(row.essay) + "\n")
    loaded = load_essays(split, min_paras=cfg.data.min_paras, max_paras=cfg.data.max_paras)
    if loaded != [row.essay for row in selected]:
        raise AssertionError("Saved split differs from the selected original records")
    for row in selected:
        if file_hash(Path(row.provenance["raw_source"])) != row.provenance["raw_source_sha256"]:
            raise ValueError("A selected raw record changed during split creation")
    manifest = {"stage": 2, "ok": True, "split": str(split.resolve()), "split_sha256": file_hash(split),
                "essay_count": len(loaded), "paragraph_distribution": dict(sorted(Counter(len(row.paragraphs) for row in loaded).items())),
                "rubric_keys": contract["rubric_keys"], "reproducible_selection": True,
                "llm_api_calls": 0, "model_inference_calls": 0, "audit": audit, "sampling": sampling}
    write_json(run / "split_manifest.json", manifest)
    write_json(split.with_suffix(".manifest.json"), manifest)
    with (run / "excluded.jsonl").open("x", encoding="utf-8") as handle:
        for row in excluded:
            handle.write(json_text(row) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key not in ("audit", "sampling", "rubric_keys")}
                     | {"run_dir": str(run), "eligible_count": audit["eligible_count"],
                        "training_overlap_rows": audit["training_overlap_rows"],
                        "strata_available": sampling["available"], "strata_selected": sampling["quotas"],
                        "cutpoints": sampling["cutpoints"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
