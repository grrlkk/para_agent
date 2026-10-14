"""Project-owned genre metadata derived with the unchanged FEAK-Agent heading rules."""
from collections import Counter, defaultdict
import copy
import importlib
import json
from pathlib import Path
import sys

from para_agent.artifacts import file_hash, json_text, write_json
from para_agent.config import project_path
from para_agent.data.loader import iter_source, load_essays, original_paragraphs, source_question_essay, text_hash


def validate_metadata_override(base_path: Path, override_path: Path):
    base = json.loads(base_path.read_text(encoding="utf-8"))
    override = json.loads(override_path.read_text(encoding="utf-8"))
    if override.get("provenance", {}).get("base_metadata_sha256") != file_hash(base_path):
        raise ValueError("Supplemental genres belong to a different original metadata file")
    if override.get("allowed_labels") != base["allowed_labels"]:
        raise ValueError("Supplemental genres must preserve the existing allowed labels")
    if any(override.get("questions", {}).get(key) != value for key, value in base["questions"].items()):
        raise ValueError("Supplemental genres must preserve every existing question entry")
    if any(value.get("genre") not in base["allowed_labels"] for value in override["questions"].values()):
        raise ValueError("Supplemental genres contain an unknown label")
    return override


def merge_heading_evidence(base, heading_sets, observations):
    """Retain existing entries; reject ambiguous or conflicting heading evidence."""
    result = copy.deepcopy(base)
    added = {}
    for question_hash, headings in sorted(observations.items()):
        if len(headings) != 1:
            raise ValueError(f"Conflicting rubric heading sets for question {question_hash}")
        chosen = next(iter(headings))
        matches = [genre for genre, expected in heading_sets.items() if len(chosen) == 8 and set(chosen) == expected]
        if len(matches) != 1:
            raise ValueError(f"Existing rules cannot identify the genre for question {question_hash}")
        genre = matches[0]
        if question_hash in base["questions"]:
            if base["questions"][question_hash]["genre"] != genre:
                raise ValueError(f"Observed genre conflicts with existing metadata for {question_hash}")
            continue
        entry = {"genre": genre, "method": "offline_feedback_heading_set",
                 "rows": sum(headings.values()), "distinct_heading_sets": 1}
        result["questions"][question_hash] = entry
        added[question_hash] = entry
    return result, added


def prepare_metadata(config, split_manifest: Path):
    from .scorer_adapter import load_scorer_settings

    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != 2 or manifest.get("ok") is not True:
        raise ValueError("Use a successful stage 2 split manifest")
    split = Path(manifest["split"])
    if file_hash(split) != manifest["split_sha256"]:
        raise ValueError("Pilot split changed after its provenance was recorded")
    essays = load_essays(split, min_paras=config.data.min_paras, max_paras=config.data.max_paras)
    provenance = {row["essay_id"]: row for row in manifest["sampling"]["selected"]}
    if len(essays) != manifest["essay_count"] or set(provenance) != {row.essay_id for row in essays}:
        raise ValueError("Split records differ from the recorded selection")
    for essay in essays:
        info = provenance[essay.essay_id]
        raw_path = Path(info["raw_source"])
        if file_hash(raw_path) != info["raw_source_sha256"]:
            raise ValueError("Selected raw source changed")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        if essay.prompt != raw["essay_question"]["prompt"] or essay.paragraphs != original_paragraphs(raw["essay_answer"]["text"]):
            raise ValueError("Selected prompt or paragraphs differ from the original source")

    # Start from the original metadata even if the config already points at a previous supplement.
    settings = load_scorer_settings(config.scorer.model_copy(update={"metadata_dir": None}))
    base_path = settings["paths"]["metadata"] / "genres.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    source = Path(manifest["audit"]["source_paths"]["test_jsonl"])
    source_hash = manifest["audit"]["source_sha256"]["test_jsonl"]
    if file_hash(source) != source_hash:
        raise ValueError("Audited test source changed")

    root = Path(config.scorer.repository_path).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    policy = importlib.import_module("verak.v3.data_policy")
    rule_path = Path(policy.__file__).resolve()
    if not rule_path.is_relative_to(root):
        raise ValueError("Imported genre rules are outside the configured FEAK-Agent repository")
    targets = {text_hash(essay.prompt) for essay in essays}
    observations, source_lines = defaultdict(Counter), defaultdict(list)
    for line_number, row in iter_source(source):
        question, _ = source_question_essay(row)
        question_hash = text_hash(question)
        if question_hash in targets:
            # Inspect only rubric names; feedback contents and gold scores never enter model inputs.
            headings = policy.feedback_headings(row["assistant"])
            observations[question_hash][headings] += 1
            source_lines[question_hash].append(line_number)
    if set(observations) != targets:
        raise ValueError("Selected questions are missing from the audited source")
    supplemented, added = merge_heading_evidence(base, policy.HEADING_SETS, observations)
    metadata_provenance = {"base_metadata": str(base_path), "base_metadata_sha256": file_hash(base_path),
                           "split_manifest": str(split_manifest.resolve()), "split_manifest_sha256": file_hash(split_manifest),
                           "split_sha256": manifest["split_sha256"], "test_source": str(source),
                           "test_source_sha256": source_hash, "heading_rule_source": str(rule_path),
                           "heading_rule_sha256": file_hash(rule_path), "llm_api_calls": 0}
    supplemented.update(source="Existing FEAK-Agent genres plus audited pilot rubric headings",
                        source_sha256=file_hash(split_manifest), provenance=metadata_provenance)
    identity = text_hash(json_text(supplemented))
    directory = project_path("data/scorer_metadata") / identity[:16]
    output = directory / "genres.json"
    report = {"metadata_dir": str(directory), "genres_sha256": identity,
              "base_question_count": len(base["questions"]), "added_question_count": len(added),
              "total_question_count": len(supplemented["questions"]), "selected_essay_count": len(essays),
              "added_questions": added, "source_lines": dict(source_lines),
              "coverage": [{"essay_id": row.essay_id, "question_sha256": text_hash(row.prompt),
                            "genre": supplemented["questions"][text_hash(row.prompt)]["genre"]} for row in essays],
              "provenance": metadata_provenance}
    directory.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text(encoding="utf-8")) != supplemented:
            raise ValueError("Refusing to overwrite different supplemental metadata")
    else:
        write_json(output, supplemented)
    report["genres_sha256"] = file_hash(output)
    report_path = directory / "provenance.json"
    if report_path.exists():
        if json.loads(report_path.read_text(encoding="utf-8")) != report:
            raise ValueError("Refusing to overwrite different metadata provenance")
    else:
        write_json(report_path, report)
    validate_metadata_override(base_path, output)
    return report
