"""Evaluate saved whole essays without rerunning any revision agents."""
import hashlib
import json
from pathlib import Path

import yaml

from para_agent.artifacts import file_hash, write_json
from para_agent.config import Config
from para_agent.data.loader import load_essays
from .edit_metrics import edit_metrics

CONDITIONS = ("whole_once", "incremental")


def load_completed_run(run: Path):
    snapshot = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    provenance = snapshot.pop("provenance")
    cfg = Config.model_validate(snapshot)
    acceptance = json.loads((run / "acceptance.json").read_text(encoding="utf-8"))
    if acceptance.get("ok") is not True or acceptance.get("mock") is not False:
        raise ValueError("Evaluation requires a completed real run")
    split = Path(provenance["input_file"])
    if file_hash(split) != provenance["input_sha256"]:
        raise ValueError("The experiment split has changed")
    available = {e.essay_id: e for e in load_essays(
        split, min_paras=cfg.data.min_paras, max_paras=cfg.data.max_paras)}
    ids = provenance["essay_ids"]
    if len(set(ids)) != len(ids) or len(ids) != acceptance["essays"]:
        raise ValueError("Run essay IDs disagree with acceptance record")
    essays = [available[essay_id] for essay_id in ids]
    finals, hashes = {}, {str(split): file_hash(split)}
    for name in ("config.yaml", "acceptance.json", "llm_calls.jsonl", "diagnostic_calls.jsonl"):
        hashes[str(run / name)] = file_hash(run / name)
    for condition in CONDITIONS:
        actual_ids = {p.parent.name for p in (run / condition).glob("*/final.json")}
        if actual_ids != set(ids):
            raise ValueError("Both conditions must contain exactly the same completed essays")
        for essay in essays:
            path = run / condition / essay.essay_id / "final.json"
            final = json.loads(path.read_text(encoding="utf-8"))
            if (final["essay_id"] != essay.essay_id or final["condition"] != condition
                    or final["mock"] is not False or not final["final_text"].strip()
                    or final["final_text"] != "\n\n".join(final["Y"])):
                raise ValueError(f"Inconsistent final artifact: {path}")
            finals[condition, essay.essay_id] = final
            hashes[str(path)] = file_hash(path)
    protected_path = run / "protected_sources.json"
    if protected_path.exists():
        protected = json.loads(protected_path.read_text(encoding="utf-8"))
        if any(file_hash(Path(path)) != digest for path, digest in protected.items()):
            raise ValueError("A protected legacy source or KB has changed")
        hashes.update(protected)
    return cfg, essays, finals, provenance, acceptance, hashes


def score_essays(essays, finals, scorer, rubric_keys, output: Path, *, progress=None):
    rows = []
    for essay in essays:
        original = "\n\n".join(essay.paragraphs)
        baseline = baseline_greedy = None
        for condition in ("original", *CONDITIONS):
            final = None if condition == "original" else finals[condition, essay.essay_id]
            text = original if final is None else final["final_text"]
            measurement=None
            if hasattr(scorer,'score_for_essay'):
                measurement=scorer.score_for_essay(text,essay.prompt,essay_id=essay.essay_id)
                scores,rationales=measurement['scores'],None
            else:
                scores, rationales = scorer.score(text, essay.prompt)
            if set(scores) != set(rubric_keys):
                raise ValueError("Final scorer keys differ from the verified rubric contract")
            if condition == "original":
                # Human original_scores are sampling labels, never this baseline.
                baseline = scores
                baseline_greedy=measurement['greedy_scores'] if measurement else None
            row = {"essay_id": essay.essay_id, "condition": condition,
                   "paragraphs": len(essay.paragraphs) if final is None else len(final["Y"]),
                   **scores, "total": sum(scores.values()),
                   **{f"delta_{key}": scores[key] - baseline[key] for key in rubric_keys},
                   "delta_total": sum(scores.values()) - sum(baseline.values()),
                   **edit_metrics(original, text)}
            if measurement:
                greedy=measurement['greedy_scores']
                if set(greedy)!=set(rubric_keys):
                    raise ValueError('Greedy scorer keys differ from verified contract')
                row.update(kanana_method=measurement['method'],kanana_n=measurement['n'],
                    kanana_seed=measurement['seed'],metric_role=measurement['metric_role'],
                    evaluation_policy_version=measurement['policy_version'],
                    **{f'greedy_{k}':v for k,v in greedy.items()},greedy_total=sum(greedy.values()),
                    **{f'greedy_delta_{k}':greedy[k]-baseline_greedy[k] for k in rubric_keys},
                    greedy_delta_total=sum(greedy.values())-sum(baseline_greedy.values()))
            write_json(output / condition / essay.essay_id / "score.json", {
                **row, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(essay.prompt.encode()).hexdigest(),
                "rubric_rationales": rationales,
                **({'smoothed_sampling':measurement['sampling']} if measurement else {})})
            rows.append(row)
            if progress:
                progress(row)
    return rows
