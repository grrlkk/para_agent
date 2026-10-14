#!/usr/bin/env python
"""Post-report stage 1: saved-text sensitivity analysis; local Kanana only.

This does not change revision, feature extraction, scoring, or KB code. Sentence
alignments below were reviewed against the three requested saved outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from para_agent.artifacts import append_jsonl, create_run, file_hash, write_json
from para_agent.diagnose.contracts import read_legacy_contract
from para_agent.diagnose.scorer_adapter import KananaScorerAdapter
from para_agent.eval.final_score import load_completed_run
from para_agent.text import sentences

SOURCE = ROOT / "runs/20261014T111644.364558Z_61faf37a02"
EVALUATION = ROOT / "runs/20261014T112554.540206Z_b6294aab18"
SPLIT_MANIFEST = ROOT / "runs/20261014T091043.217356Z_1f33a9eca6/split_manifest.json"
FRIEND = "14-2-E5-N-0C-F-0084"
LAW = "14-2-M1-C-0C-E-1504"
MARK = re.compile(r"#@([^#\r\n]+)#")
NEUTRAL = {"이름": "가명", "비속어": "욕설"}
# paragraph (1-based), first/last original sentence, first/last revised sentence
ALIGNMENTS = {
    (LAW, "whole_once"): [(1, 5, 5, 5, 5), (2, 3, 3, 3, 3)],
    (FRIEND, "whole_once"): [
        *[(1, i, i, i, i) for i in range(1, 6)],
        (2, 1, 1, 1, 1), (2, 2, 2, 2, 2), (2, 3, 3, 3, 4),
        (2, 4, 4, 5, 6), (3, 2, 2, 2, 2), (3, 3, 3, 3, 4)],
    (FRIEND, "incremental"): [
        (1, 3, 3, 3, 3), (2, 3, 3, 3, 3), (3, 2, 2, 2, 2), (3, 3, 3, 3, 4)],
}


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_hashes(hashes):
    changed = [p for p, digest in hashes.items() if file_hash(Path(p)) != digest]
    if changed:
        raise ValueError(f"Changed protected inputs: {changed}")


def sentence_span(paragraphs, p, first, last):
    text = paragraphs[p - 1]
    parts = sentences(text)
    spans, cursor = [], 0
    for part in parts:
        start = text.index(part, cursor)
        spans.append((start, start + len(part)))
        cursor = start + len(part)
    assert 1 <= first <= last <= len(spans)
    offset = sum(len(part) + 2 for part in paragraphs[:p - 1])
    return offset + spans[first - 1][0], offset + spans[last - 1][1]


def replace_spans(text, replacements):
    previous = len(text) + 1
    for start, end, replacement in sorted(replacements, reverse=True):
        assert 0 <= start <= end < previous
        text = text[:start] + replacement + text[end:]
        previous = start
    return text


def prepare():
    cfg, essays, finals, provenance, acceptance, hashes = load_completed_run(SOURCE)
    manifest = read_json(SPLIT_MANIFEST)
    contract = read_legacy_contract(cfg.scorer.package_path)
    hashes.update({str(p): file_hash(p) for p in [
        SPLIT_MANIFEST, EVALUATION / "results.csv", EVALUATION / "acceptance.json",
        *EVALUATION.glob("*/*/score.json"),
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
        *sorted((ROOT / "prompts").glob("*.md")),
        *sorted((ROOT / "configs").glob("*.yaml")),
    ]})
    raw_sources = {r["essay_id"]: r for r in manifest["sampling"]["selected"]}
    for essay in essays:
        row = raw_sources[essay.essay_id]
        assert file_hash(Path(row["raw_source"])) == row["raw_source_sha256"]
        hashes[row["raw_source"]] = row["raw_source_sha256"]
    run = create_run(ROOT / "runs", ROOT, cfg.model_dump(), {
        "followup_stage": 1, "kind": "local_kanana_sensitivity", "source_run": str(SOURCE),
        "source_evaluation": str(EVALUATION), "real_llm_api_calls": 0,
        "dataset_role": "development_only", "essay_ids": [e.essay_id for e in essays],
        "future_evaluation_exclude_ids": list(raw_sources),
    })
    write_json(run / "input_hashes.json", hashes)
    write_json(run / "rubric_contract.json", contract)
    (run / "analysis_script.py").write_bytes(Path(__file__).read_bytes())
    texts, variants, cases, text_audit, edits = {}, [], [], [], []

    def add(essay, name, kind, text, *, force=False, **extra):
        key = (essay.prompt, text)
        if force or key not in texts:
            case_id = f"case_{len(cases) + 1:04d}"
            path = run / "texts" / f"{case_id}.txt"
            path.parent.mkdir(exist_ok=True)
            path.write_text(text, encoding="utf-8")
            cases.append({"case_id": case_id, "essay_id": essay.essay_id,
                          "prompt": essay.prompt, "prompt_sha256": sha(essay.prompt),
                          "text_path": str(path.relative_to(run)), "text_sha256": sha(text),
                          "characters": len(text), "first_variant": name,
                          "identical_input_repeat": force})
            if not force:
                texts[key] = case_id
        else:
            case_id = texts[key]
        variants.append({"essay_id": essay.essay_id, "variant": name, "kind": kind,
                         "case_id": case_id, **extra})
        return case_id

    # All canonical baselines are explicitly reconstructed exactly as evaluate.py did.
    for essay in essays:
        original = "\n\n".join(essay.paragraphs)
        score = read_json(EVALUATION / "original" / essay.essay_id / "score.json")
        assert score["text_sha256"] == sha(original)
        assert score["prompt_sha256"] == sha(essay.prompt)
        add(essay, "original_joined", "baseline", original, stored_scores=score)
    for essay in essays:
        for condition in ("whole_once", "incremental"):
            score = read_json(EVALUATION / condition / essay.essay_id / "score.json")
            assert score["text_sha256"] == sha(finals[condition, essay.essay_id]["final_text"])
            assert score["prompt_sha256"] == sha(essay.prompt)
            if (essay.essay_id, condition) in ALIGNMENTS:
                add(essay, condition, "revised_baseline", finals[condition, essay.essay_id]["final_text"],
                    condition=condition, stored_scores=score)

    for essay in essays:
        original = "\n\n".join(essay.paragraphs)
        raw_path = Path(raw_sources[essay.essay_id]["raw_source"])
        raw = read_json(raw_path)["essay_answer"]["text"]
        assert sha(raw) == raw_sources[essay.essay_id]["original_text_sha256"]
        add(essay, "original_raw", "raw_comparison", raw)
        marks = [m.group(0) for m in MARK.finditer(original)]
        unknown = {m.group(1) for m in MARK.finditer(original)} - NEUTRAL.keys()
        if unknown:
            raise ValueError(f"No invented neutral replacement for unknown tags: {unknown}")
        transformations = {
            "a_delete_markers": MARK.sub("", original),
            "b_neutral_markers": MARK.sub(lambda m: NEUTRAL[m.group(1)], original),
            "c_single_newline": "\n".join(essay.paragraphs),
            "d_tidy_sentence_whitespace": "\n\n".join(
                re.sub(r"([.!?。！？])[ \t]+(?=\S)", r"\1 ", p.rstrip(" \t\r\n"))
                for p in essay.paragraphs),
        }
        for name, text in transformations.items():
            if name.startswith(("c_", "d_")):
                assert re.sub(r"\s+", "", text) == re.sub(r"\s+", "", original)
            add(essay, name, "sensitivity", text, changed=text != original,
                marker_count=len(marks), base_variant="original_joined")
        text_audit.append({"essay_id": essay.essay_id, "raw_source": str(raw_path),
                           "raw_sha256": sha(raw), "joined_sha256": sha(original),
                           "raw_equals_joined": raw == original,
                           "raw_cr_count": raw.count("\r"), "raw_lf_count": raw.count("\n"),
                           "raw_characters": len(raw), "joined_characters": len(original),
                           "paragraphs": len(essay.paragraphs), "markers": marks,
                           "original_evaluation_used_joined_text": True,
                           "both_final_evaluations_used_joined_text": True})

    for (essay_id, condition), pairs in ALIGNMENTS.items():
        essay = next(e for e in essays if e.essay_id == essay_id)
        original = "\n\n".join(essay.paragraphs)
        final = finals[condition, essay_id]
        revised = final["final_text"]
        all_reverse = []
        for number, (p, a1, a2, b1, b2) in enumerate(pairs, 1):
            a_start, a_end = sentence_span(essay.paragraphs, p, a1, a2)
            b_start, b_end = sentence_span(final["Y"], p, b1, b2)
            before, after = original[a_start:a_end], revised[b_start:b_end]
            assert before != after
            edit_id = f"{condition}_p{p}_s{a1}_{a2}"
            edits.append({"essay_id": essay_id, "condition": condition, "edit_id": edit_id,
                          "number": number, "paragraph": p, "before_sentence_ids": [a1, a2],
                          "after_sentence_ids": [b1, b2], "before": before, "after": after,
                          "original_span": [a_start, a_end], "revised_span": [b_start, b_end]})
            all_reverse.append((b_start, b_end, before))
            add(essay, f"revert_{edit_id}", "single_revert",
                replace_spans(revised, [(b_start, b_end, before)]),
                condition=condition, edit_id=edit_id, base_variant=condition)
            # Original-context control exposes nonadditive/context-dependent effects.
            add(essay, f"apply_{edit_id}", "single_apply",
                replace_spans(original, [(a_start, a_end, after)]),
                condition=condition, edit_id=edit_id, base_variant="original_joined")
        all_reverted = replace_spans(revised, all_reverse)
        assert re.sub(r"\s+", "", all_reverted) == re.sub(r"\s+", "", original)
        add(essay, f"all_sentences_reverted_{condition}", "all_revert_control", all_reverted,
            condition=condition, equals_original=all_reverted == original)
        restored_spacing = "\n\n".join(
            x[:len(x) - len(x.lstrip())] + y.strip() + x[len(x.rstrip()):]
            for x, y in zip(essay.paragraphs, final["Y"]))
        if restored_spacing != revised:
            add(essay, f"restore_edge_whitespace_{condition}", "spacing_control", restored_spacing,
                condition=condition, base_variant=condition)

    # Three uncached measurements of each of the eight baseline texts.
    for essay in essays:
        for repeat in (2, 3):
            add(essay, f"repeat_original_{repeat}", "repeat", "\n\n".join(essay.paragraphs),
                force=True, base_variant="original_joined", repeat=repeat)
        for condition in ("whole_once", "incremental"):
            if (essay.essay_id, condition) in ALIGNMENTS:
                for repeat in (2, 3):
                    add(essay, f"repeat_{condition}_{repeat}", "repeat",
                        finals[condition, essay.essay_id]["final_text"],
                        force=True, base_variant=condition, repeat=repeat)

    write_json(run / "text_assembly_audit.json", text_audit)
    write_json(run / "sentence_changes.json", edits)
    write_json(run / "variants.json", variants)
    write_json(run / "cases.json", cases)
    write_json(run / "protocol.json", {
        "followup_stage": 1, "source_run": str(SOURCE), "source_evaluation": str(EVALUATION),
        "real_llm_api_calls": 0, "feature_extractor_calls": 0,
        "local_scorer_calls_planned": len(cases), "logical_variants": len(variants),
        "manual_sentence_change_groups": len(edits), "neutral_replacements": NEUTRAL,
        "cache_policy": "All actual scorer calls use_cache=False; exact duplicate variants share a score except repeat controls",
        "revert_policy": "One original sentence per intervention; a 1-to-2 split is reverted as a block; every other output character is fixed",
        "sentence_alignment": "Explicit alignment manually checked for these three saved outputs; not a production diff feature",
        "d_policy": "Fixed double-newline paragraph boundaries; strip paragraph-end whitespace and reduce whitespace after sentence-final punctuation to one space; leave all non-whitespace characters unchanged",
        "marker_caveat": "Deletion can leave particles dangling; category substitution cannot recover hidden words or guarantee original meaning. Treat a/b as marker sensitivity, not proven meaning-preserving variants.",
        "dataset_role": "development_only", "future_evaluation_exclude_ids": list(raw_sources),
        "source_model_fingerprint": read_json(EVALUATION / "acceptance.json")["model_fingerprint"],
    })
    check_hashes(hashes)
    print(json.dumps({"analysis_run": str(run), "local_scorer_calls_planned": len(cases),
                      "sentence_changes": len(edits), "real_llm_api_calls": 0}, ensure_ascii=False))


def score(run):
    protocol = read_json(run / "protocol.json")
    hashes = read_json(run / "input_hashes.json")
    check_hashes(hashes)
    cfg, *_ = load_completed_run(Path(protocol["source_run"]))
    contract = read_json(run / "rubric_contract.json")
    cfg.scorer.cache_dir = str(run / "scorer_cache")
    scorer = KananaScorerAdapter(cfg.scorer, contract["rubric_keys"], log_path=run / "scorer_calls.jsonl")
    cases = read_json(run / "cases.json")
    result_path = run / "scores.jsonl"
    if result_path.exists():
        raise ValueError("Refuse to overwrite or mix an earlier scoring attempt")
    report = {"ok": False, "followup_stage": 1, "real_llm_api_calls": 0,
              "feature_extractor_calls": 0, "scored_cases": 0, "cache_hits": 0}
    started = time.perf_counter()
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu"] = scorer.settings["scorer"]["gpu"]
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the unchanged scorer configuration")
        for index, case in enumerate(cases, 1):
            # Preserve raw CRLF exactly; text-mode reads would normalize newlines.
            text = (run / case["text_path"]).read_bytes().decode("utf-8")
            assert sha(text) == case["text_sha256"]
            before = time.perf_counter()
            scores, _ = scorer.score(text, case["prompt"], use_cache=False)
            if scorer.backend.fingerprint != protocol["source_model_fingerprint"]:
                raise ValueError("Scorer fingerprint changed")
            row = {**case, "scores": scores, "total": sum(scores.values()),
                   "latency_s": time.perf_counter() - before}
            append_jsonl(result_path, row)
            report["scored_cases"] = index
            print(json.dumps({"progress": f"{index}/{len(cases)}", "case_id": case["case_id"],
                              "essay_id": case["essay_id"], "variant": case["first_variant"],
                              "total": row["total"]}, ensure_ascii=False), flush=True)
        check_hashes(hashes)
        calls = [json.loads(line) for line in (run / "scorer_calls.jsonl").read_text().splitlines()]
        report["cache_hits"] = sum(call["result"]["cache_hit"] for call in calls)
        assert report["cache_hits"] == 0 and len(calls) == len(cases)
        assert all(call["use_cache"] is False and call["truncated"] is False for call in calls)
        report.update(ok=True, all_input_hashes_unchanged=True, truncated_inputs=0,
                      model_fingerprint=scorer.backend.fingerprint,
                      protected_legacy_and_kb_files=len(read_json(SOURCE / "protected_sources.json")),
                      min_input_tokens=min(c["result"]["input_tokens"] for c in calls),
                      max_input_tokens=max(c["result"]["input_tokens"] for c in calls))
    finally:
        scorer.close()
        report["latency_s"] = time.perf_counter() - started
        write_json(run / "acceptance.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("prepare")
    scoring = sub.add_parser("score")
    scoring.add_argument("--analysis-run", type=Path, required=True)
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.action == "prepare":
        prepare()
    else:
        score(args.analysis_run.resolve())


if __name__ == "__main__":
    main()
