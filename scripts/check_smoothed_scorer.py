#!/usr/bin/env python
"""Stage 1.5 experiment: preregister, sample, and compare local Kanana estimators."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import shutil

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from para_agent.artifacts import append_jsonl, create_run, file_hash, write_json
from para_agent.config import Config, load_config
from para_agent.data.loader import load_heldout_candidates, stratified_sample, text_hash, iter_source, source_question_essay
from para_agent.diagnose.contracts import read_legacy_contract
from para_agent.diagnose.scorer_adapter import KananaScorerAdapter, load_scorer_settings
from para_agent.diagnose.scorer_metadata import merge_heading_evidence, validate_metadata_override
from para_agent.diagnose.smoothed_scorer import SmoothedKananaAdapter, essay_seed, summarize_samples

STAGE1 = ROOT / "runs/20261014T121907.365037Z_32be8ba77f"
PILOT = ROOT / "runs/20261014T091043.217356Z_1f33a9eca6/split_manifest.json"
SOURCE_RUN = ROOT / "runs/20261014T111644.364558Z_61faf37a02"
FRIEND = "14-2-E5-N-0C-F-0084"
SELECTION_SEED = 15042


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def check_protected(run):
    changed = [p for p, digest in read(run / "protected_sources.json").items() if file_hash(Path(p)) != digest]
    if changed:
        raise ValueError(f"Protected source changed: {changed}")


def prepare():
    from transformers import AutoTokenizer
    cfg = load_config(ROOT / "configs/pilot_haiku.yaml")
    contract = read_legacy_contract(cfg.scorer.package_path)
    pilot = read(PILOT)
    excluded_ids = {r["essay_id"] for r in pilot["sampling"]["selected"]}
    excluded_texts = {r["whitespace_ignored_sha256"] for r in pilot["sampling"]["selected"]}
    settings = load_scorer_settings(cfg.scorer)
    sys.path.insert(0, cfg.scorer.repository_path)
    native_module = importlib.import_module("verak.v3.score.kanana")
    tokenizer = AutoTokenizer.from_pretrained(str(settings["paths"]["scorer_adapter"]), local_files_only=True)

    def token_count(question, text):
        chat = [{"role": "system", "content": native_module.SYSTEM_PROMPT},
                {"role": "user", "content": f"질문: {question}\n에세이: {text}"}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return len(tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"])

    # Fix the rule before any new model inference. Only the cohort metadata is populated below.
    preregistration = {
        "stage": "1.5", "selection_seed": SELECTION_SEED, "sampling_seed": SELECTION_SEED,
        "temperature": 1.0, "n_checkpoints": [8, 16, 32, 64], "max_n": 64,
        "primary_method": "integer_mean", "primary_n": 32, "primary_scale": "sum_of_8_rubrics",
        "primary_invariance": {"lf": "all five development essays, double LF versus single LF",
                               "contraction": "friend whole_once p1.s5: 못하였더라도 → 못했더라도",
                               "each_family_max_abs_delta_at_most": 1.0},
        "primary_agreement": "100 canonical double-LF originals: integer_mean N=32 total Spearman >= native greedy-expected total Spearman",
        "decision": "All three pass: recommend smoothed primary pending approval. Otherwise recommend Kanana auxiliary and independent judge pairwise/meaning preservation primary.",
        "secondary_methods": ["expected_mean N=8,16,32,64", "integer_mean N=8,16,64"],
        "no_posthoc_method_or_n_selection": True, "no_score_based_resampling": True,
        "sample_policy": "full vocabulary temperature=1; top_k=0 top_p=1; first 64 valid first-line eight-integer outputs; max 256 attempts; log every rejected line; no replacement scores",
        "coupling": "same essay seed for all text variants; nested sample prefixes for N; no seed selection after results",
        "cohort": "100 scorer-held-out test essays, original boundaries, 3–6 paragraphs, human-total tercile quotas 34/33/33; exclude pilot 10 and whitespace-equivalent texts; both raw/canonical inputs <=3072 tokens before sampling",
        "primary_input_assembly": "\\n\\n.join(original_paragraphs)",
        "raw_reference": "Score original raw strings by both greedy and both sampled estimators, secondary only",
        "device_policy": "place all variants of an essay on the same physical GPU; GPU0/GPU1; verify identical model/LoRA and cross-device control",
        "batch_policy": "batch 8 generation, batch 8 expected; benchmark checks selected-position head matches full head; check batch-vs-single expected error <=0.02 per rubric and <=0.05 total; use expected batch 1 if exceeded",
        "real_llm_api_calls": 0, "external_api_cost_usd": 0,
        "existing_evaluation_settings_changed": False,
    }
    run = create_run(ROOT / "runs", ROOT, cfg.model_dump(), {
        "stage": "1.5", "kind": "local_smoothed_kanana_validation", "source_stage1": str(STAGE1),
        "real_llm_api_calls": 0, "existing_evaluation_settings_changed": False})
    write_json(run / "preregistration.json", preregistration)
    shutil.copy2(ROOT / "configs/smoothed_scoring.yaml", run / "sampling_settings.yaml")
    print(f"run: {run}", flush=True)
    protected = read(SOURCE_RUN / "protected_sources.json")
    protected.update({str(p): file_hash(p) for p in [PILOT, ROOT / "configs/pilot_haiku.yaml",
        ROOT / "configs/default.yaml", ROOT / "configs/smoothed_scoring.yaml", *sorted((ROOT / "prompts").glob("*.md")),
        *sorted((ROOT / "src").rglob("*.py")), Path(__file__)]})
    write_json(run / "protected_sources.json", protected)
    write_json(run / "rubric_contract.json", contract)
    candidates, audit, excluded = load_heldout_candidates(cfg.data.source, contract["rubric_keys"], min_paras=3, max_paras=6)
    eligible, extra_excluded = [], []
    for candidate in candidates:
        e = candidate.essay
        if e.essay_id in excluded_ids or candidate.provenance["whitespace_ignored_sha256"] in excluded_texts:
            extra_excluded.append({"essay_id": e.essay_id, "reason": "pilot_10_or_equivalent"})
            continue
        raw = read(Path(candidate.provenance["raw_source"]))["essay_answer"]["text"]
        canonical = "\n\n".join(e.paragraphs)
        lengths = {"canonical": token_count(e.prompt, canonical), "raw": token_count(e.prompt, raw)}
        if max(lengths.values()) > settings["scorer"]["max_input_tokens"]:
            extra_excluded.append({"essay_id": e.essay_id, "reason": "input_too_long", "input_tokens": lengths})
            continue
        eligible.append(candidate)
    selected, sampling = stratified_sample(eligible, count=100, seed=SELECTION_SEED)
    split_path = ROOT / "data/splits" / f"smoothing_calibration_100_{run.name}.jsonl"
    split_path.parent.mkdir(exist_ok=True)
    for c in selected:
        append_jsonl(split_path, c.essay)
    manifest = {"stage": "1.5", "dataset_role": "scorer_calibration_not_final_evaluation", "essay_count": 100,
                "split": str(split_path), "split_sha256": file_hash(split_path),
                "sampling": sampling, "source_audit": audit, "extra_exclusions": extra_excluded,
                "excluded_pilot_ids": sorted(excluded_ids), "model_inference_calls_during_selection": 0}
    write_json(run / "calibration_manifest.json", manifest)
    (run / "calibration_100_ids.txt").write_text("\n".join(c.essay.essay_id for c in selected) + "\n")
    exclusion_record = {"reason": "Pilot development plus stage-1.5 scorer calibration; exclude from stage-5 evaluation",
                        "pilot_ids": sorted(excluded_ids), "calibration_ids": [c.essay.essay_id for c in selected],
                        "exclude_essay_ids": sorted(excluded_ids | {c.essay.essay_id for c in selected}),
                        "exclude_whitespace_ignored_sha256": sorted(excluded_texts | {c.provenance["whitespace_ignored_sha256"] for c in selected}),
                        "manifest": str(run / "calibration_manifest.json"), "manifest_sha256": file_hash(run / "calibration_manifest.json")}
    exclusion_path = ROOT / "data/splits" / f"evaluation_exclusions_stage1_5_{run.name}.json"
    write_json(exclusion_path, exclusion_record)
    write_json(run / "future_evaluation_exclusions.json", {**exclusion_record, "persistent_record": str(exclusion_path)})

    source_cases = {r["case_id"]: r for r in read(STAGE1 / "cases.json")}
    source_variants = read(STAGE1 / "variants.json")
    essay_questions = {r["essay_id"]: r["prompt"] for r in source_cases.values()}
    records, aliases, unique = [], [], {}

    def add(eid, group, label, question, text, **extra):
        key = (text_hash(question), text_hash(text))
        if key not in unique:
            case_id = f"input_{len(records) + 1:04d}"
            path = run / "texts" / f"{case_id}.txt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(text, encoding="utf-8")
            records.append({"case_id": case_id, "essay_id": eid, "group": group, "prompt": question,
                            "prompt_sha256": key[0], "text_sha256": key[1], "text_path": str(path.relative_to(run)),
                            "input_tokens": token_count(question, text), "seed": essay_seed(eid, SELECTION_SEED)})
            unique[key] = case_id
        aliases.append({"essay_id": eid, "group": group, "label": label, "case_id": unique[key], **extra})

    for v in source_variants:
        keep = v["kind"] in ("baseline", "raw_comparison") or v["variant"] == "c_single_newline"
        keep |= v["kind"] == "single_apply" and v["essay_id"] == FRIEND
        if keep:
            c = source_cases[v["case_id"]]
            text = (STAGE1 / c["text_path"]).read_bytes().decode()
            assert text_hash(text) == c["text_sha256"]
            add(v["essay_id"], "development", v["variant"], c["prompt"], text,
                source_case=c["case_id"], kind=v["kind"])
    for r in lines(STAGE1 / "targeted_checks/results.jsonl"):
        if r["name"].startswith("law_factors_"):
            path = STAGE1 / "targeted_checks" / r["text_path"] if "text_path" in r else STAGE1 / source_cases[r["existing_case_id"]]["text_path"]
            text = path.read_bytes().decode()
            assert text_hash(text) == r["text_sha256"]
            add(r["essay_id"], "development", r["name"], r["prompt"], text, kind="law_factor")
    for c in selected:
        e = c.essay
        raw = read(Path(c.provenance["raw_source"]))["essay_answer"]["text"]
        for label, text in (("canonical", "\n\n".join(e.paragraphs)), ("raw", raw)):
            add(e.essay_id, "calibration", label, e.prompt, text, human_scores=e.original_scores)
    write_json(run / "cases.json", records)
    write_json(run / "aliases.json", aliases)

    # Supplemental genre metadata uses existing heading rules; no feedback text or scores enter prompts.
    policy = importlib.import_module("verak.v3.data_policy")
    base_settings = load_scorer_settings(cfg.scorer.model_copy(update={"metadata_dir": None}))
    base_path = base_settings["paths"]["metadata"] / "genres.json"
    base = read(base_path)
    targets = {r["prompt_sha256"] for r in records}
    observations = defaultdict(Counter)
    for _, row in iter_source(Path(cfg.data.source.test_jsonl)):
        question, _ = source_question_essay(row)
        qhash = text_hash(question)
        if qhash in targets:
            observations[qhash][policy.feedback_headings(row["assistant"])] += 1
    assert set(observations) == targets
    metadata, added = merge_heading_evidence(base, policy.HEADING_SETS, observations)
    metadata["provenance"] = {"base_metadata": str(base_path), "base_metadata_sha256": file_hash(base_path),
                              "heading_rule_source": policy.__file__, "heading_rule_sha256": file_hash(Path(policy.__file__)),
                              "real_llm_api_calls": 0, "calibration_manifest_sha256": file_hash(run / "calibration_manifest.json")}
    metadata_dir = run / "metadata"
    write_json(metadata_dir / "genres.json", metadata)
    validate_metadata_override(base_path, metadata_dir / "genres.json")
    cfg.scorer.metadata_dir = str(metadata_dir)
    write_json(run / "analysis_config.json", cfg.model_dump())

    # Work estimates depend only on token lengths, not on predictions or human labels.
    group_cases = defaultdict(list)
    for case in records:
        group_cases[case["essay_id"]].append(case)
    loads, assignments = [0, 0], {0: [], 1: []}
    workloads = {eid: sum(c["input_tokens"] + c["input_tokens"]**2 / 3072 for c in cs) for eid, cs in group_cases.items()}
    for eid in sorted(group_cases, key=lambda e: (-workloads[e], e)):
        gpu = min((0, 1), key=lambda g: loads[g])
        assignments[gpu].extend(c["case_id"] for c in group_cases[eid])
        loads[gpu] += workloads[eid]
    write_json(run / "gpu_assignments.json", {str(g): {"case_ids": assignments[g], "estimated_work": loads[g]} for g in (0, 1)})
    write_json(run / "preparation.json", {"ok": True, "unique_inputs": len(records),
        "development_inputs": sum(c["group"] == "development" for c in records),
        "calibration_inputs": sum(c["group"] == "calibration" for c in records),
        "sampled_lines_planned": len(records)*64, "native_greedy_calls_planned": len(records),
        "future_exclusion_ids": len(exclusion_record["exclude_essay_ids"]),
        "added_genre_questions": len(added), "real_llm_api_calls": 0,
        "gpu_cases": {str(g): len(assignments[g]) for g in (0, 1)},
        "preregistration_sha256": file_hash(run / "preregistration.json")})
    check_protected(run)
    print(json.dumps({"run": str(run), **read(run / "preparation.json")}, ensure_ascii=False), flush=True)


def make_native(run, gpu, log_path):
    cfg = Config.model_validate(read(run / "analysis_config.json"))
    cfg.scorer.cache_dir = str(run / f"cache_gpu{gpu}")
    contract = read(run / "rubric_contract.json")
    adapter = KananaScorerAdapter(cfg.scorer, contract["rubric_keys"], log_path=log_path)
    # Device placement is the only changed scorer inference setting.
    adapter.settings["scorer"]["gpu"] = gpu
    native = adapter._ensure_backend()
    settings_record = {
        "gpu": gpu, "scorer": adapter.settings["scorer"], "fingerprint": native.fingerprint,
        "model_paths": {key: str(adapter.settings["paths"][key]) for key in ("policy_base", "scorer_adapter")},
        "genre_metadata": adapter.metadata_provenance}
    settings_path = log_path.parent / "scorer_settings.json"
    if settings_path.exists():
        if read(settings_path) != settings_record:
            raise ValueError("Cannot resume with changed scorer settings")
    else:
        write_json(settings_path, settings_record)
    return adapter, native


def benchmark(run, gpu):
    check_protected(run)
    output = run / f"benchmark_gpu{gpu}"
    output.mkdir(exist_ok=False)
    adapter, native = make_native(run, gpu, output / "greedy_calls.jsonl")
    case = next(c for c in read(run / "cases.json") if c["essay_id"] == FRIEND and
                any(a["case_id"] == c["case_id"] and a["label"] == "original_joined" for a in read(run / "aliases.json")))
    text = (run / case["text_path"]).read_bytes().decode()
    report = {"ok": False, "gpu": gpu, "case_id": case["case_id"], "real_llm_api_calls": 0}
    started = time.perf_counter()
    try:
        scores, _ = adapter.score(text, case["prompt"], use_cache=False)
        native_result = lines(output / "greedy_calls.jsonl")[-1]["result"]
        smoother = SmoothedKananaAdapter(native, batch_size=8)
        selected_head = smoother.expected_for_lines(case["prompt"], text, [native_result["score_line"]], batch_size=1)[0]["expected"]
        projection_error = max(abs(a-b) for a,b in zip(selected_head, native_result["expected"]))
        if projection_error > 1e-8:
            raise ValueError(f"Selected-output projection does not match native full projection: {projection_error}")
        sampled = smoother.score(case["prompt"], text, n=8, seed=case["seed"],
                                 progress=lambda p: print(json.dumps({"gpu": gpu, "benchmark": p}), flush=True))
        scalar = smoother.expected_for_lines(case["prompt"], text, [r["line"] for r in sampled["samples"]], batch_size=1)
        errors = [max(abs(a-b) for a,b in zip(x["expected"], y["expected"])) for x,y in zip(sampled["samples"],scalar)]
        total_errors = [abs(sum(x["expected"])-sum(y["expected"])) for x,y in zip(sampled["samples"],scalar)]
        expected_batch = 8 if max(errors) <= .02 and max(total_errors) <= .05 else 1
        write_json(output / "sampled.json", sampled)
        report.update(ok=True, native_scores=scores, score_line=native_result["score_line"],
                      projection_max_error=projection_error, batch_vs_single_max_rubric_error=max(errors),
                      batch_vs_single_max_total_error=max(total_errors), expected_batch_size=expected_batch,
                      sampled_8_latency_s=sampled["latency_s"], rejection_count=len(sampled["rejected"]),
                      steady_projected_64_latency_s=8*sampled["latency_s"])
    finally:
        adapter.close()
        report["latency_s"] = time.perf_counter() - started
        write_json(output / "acceptance.json", report)
    check_protected(run)
    print(json.dumps(report), flush=True)


def worker(run, gpu):
    check_protected(run)
    for device in (0, 1):
        if not read(run / f"benchmark_gpu{device}/acceptance.json")["ok"]:
            raise ValueError("Both benchmarks must pass before full scoring")
    output = run / f"worker_gpu{gpu}"
    output.mkdir(exist_ok=True)
    ids = read(run / "gpu_assignments.json")[str(gpu)]["case_ids"]
    indexed = {c["case_id"]: c for c in read(run / "cases.json")}
    existing = {p.stem for p in (output / "cases").glob("*.json")}
    if not existing <= set(ids):
        raise ValueError("Saved cases do not belong to this GPU assignment")
    if (output / "acceptance.json").exists():
        raise ValueError("This worker already has a final acceptance record")
    adapter, native = make_native(run, gpu, output / "greedy_calls.jsonl")
    expected_batch = min(read(run / f"benchmark_gpu{g}/acceptance.json")["expected_batch_size"] for g in (0, 1))
    smoother = SmoothedKananaAdapter(native, batch_size=8, expected_batch_size=expected_batch)
    settings = yaml.safe_load((run / "sampling_settings.yaml").read_text(encoding="utf-8"))
    seed_settings = settings['seed']
    assert seed_settings['scheme'] == 'sha256_essay_id_v1' and seed_settings['master_seed'] == SELECTION_SEED
    assert seed_settings['digest_prefix_bytes'] == 4 and seed_settings['byte_order'] == 'big'
    assert seed_settings['modulus'] == 2**31 - 1
    assert settings['sampling']['generation_batch_size'] == 8
    assert settings['sampling']['expected_batch_size'] == expected_batch == 1
    for case in indexed.values():
        assert case['seed'] == essay_seed(case['essay_id'], seed_settings['master_seed'])
    report = {"ok": False, "gpu": gpu, "completed": len(existing), "assigned": len(ids),
              "real_llm_api_calls": 0, "expected_batch_size": expected_batch,
              "resumed_completed_inputs": len(existing), "new_inputs_completed": 0,
              "sampling_settings_sha256": file_hash(run / "sampling_settings.yaml")}
    started = time.perf_counter()
    try:
        for index, case_id in enumerate(ids, 1):
            if case_id in existing:
                saved = read(output / "cases" / f"{case_id}.json")
                case = indexed[case_id]
                for key in ('case_id', 'essay_id', 'prompt_sha256', 'text_sha256', 'seed'):
                    assert saved[key] == case[key]
                assert file_hash(run / case['text_path']) == case['text_sha256']
                assert saved['gpu'] == gpu and saved['model_fingerprint'] == native.fingerprint
                assert saved['smoothed']['seed'] == case['seed']
                assert saved['smoothed']['n'] == len(saved['smoothed']['samples']) == 64
                assert saved['smoothed']['summary'] == summarize_samples(saved['smoothed']['samples'])
                report['completed'] = index
                continue
            case = indexed[case_id]
            text = (run / case["text_path"]).read_bytes().decode()
            assert text_hash(text) == case["text_sha256"]
            scores, _ = adapter.score(text, case["prompt"], use_cache=False)
            sampled = smoother.score(case["prompt"], text, n=64, seed=case["seed"],
                                     progress=lambda p: print(json.dumps({"gpu": gpu, "case_id": case_id, **p}), flush=True))
            result = {**case, "gpu": gpu, "greedy_scores": scores, "greedy_total": sum(scores.values()),
                      "model_fingerprint": native.fingerprint, "smoothed": sampled}
            write_json(output / "cases" / f"{case_id}.json", result)
            report["completed"] = index
            report['new_inputs_completed'] += 1
            print(json.dumps({"gpu": gpu, "completed": index, "assigned": len(ids), "case_id": case_id,
                              "sampling_s": sampled["latency_s"], "elapsed_s": time.perf_counter()-started}), flush=True)
        check_protected(run)
        report.update(ok=True, all_protected_hashes_unchanged=True, cache_hits=0)
    finally:
        adapter.close()
        report["latency_s"] = time.perf_counter() - started
        write_json(output / "acceptance.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    for name in ("benchmark", "worker"):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--gpu", type=int, choices=(0,1), required=True)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    if args.action == "prepare":
        prepare()
    else:
        {"benchmark": benchmark, "worker": worker}[args.action](args.run.resolve(), args.gpu)


if __name__ == "__main__":
    main()
