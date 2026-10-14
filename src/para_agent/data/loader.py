"""Read original paragraph boundaries and sample an audited, scorer-held-out split."""
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import quantiles

from pydantic import Field, field_validator

from para_agent.schema import Schema
from para_agent.artifacts import file_hash


class Essay(Schema):
    essay_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    prompt: str = Field(min_length=1)
    paragraphs: list[str] = Field(min_length=1)
    original_scores: dict[str, float]

    @field_validator("paragraphs")
    @classmethod
    def nonempty_paragraphs(cls, value):
        if any(not part.strip() for part in value):
            raise ValueError("Original paragraph boundaries must contain nonempty paragraphs")
        return value


def load_essays(path: Path, *, min_paras: int, max_paras: int, limit: int | None = None) -> list[Essay]:
    rows = [Essay.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row.essay_id for row in rows}) != len(rows):
        raise ValueError("Essay split needs nonempty, unique essay IDs")
    if any(not min_paras <= len(row.paragraphs) <= max_paras for row in rows):
        raise ValueError("Essay paragraphs are outside configured bounds")
    return rows if limit is None else rows[:limit]


def load_mock_essays(path: Path, *, min_paras: int, max_paras: int, limit: int) -> list[Essay]:
    return load_essays(path, min_paras=min_paras, max_paras=max_paras, limit=limit)


def source_question_essay(row: dict) -> tuple[str, str]:
    """Match the verified FEAK-Agent common.extract_question_essay format exactly."""
    user = row.get("user")
    if not isinstance(user, str) or not user.startswith("질문:") or "\n에세이:" not in user:
        raise ValueError("Expected the existing 질문:/에세이: source format")
    question, text = user[len("질문:"):].split("\n에세이:", 1)
    question, text = question.removeprefix(" "), text.removeprefix(" ")
    if "\n핵심 키워드:" in text:
        text, _ = text.rsplit("\n핵심 키워드:", 1)
    if not question.strip() or not text.strip():
        raise ValueError("Empty source question or essay")
    return question, text


def original_paragraphs(text: str) -> list[str]:
    # The raw corpus stores paragraph boundaries as CR/LF, with or without blank lines.
    # Preserve all characters within each paragraph; never infer a boundary from sentences.
    return [part for part in re.split(r"\r\n|\r|\n", text) if part.strip()]


def text_hash(text: str, *, ignore_whitespace=False) -> str:
    if ignore_whitespace:
        text = re.sub(r"\s+", "", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_source(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def checked_raters(first, second, rubric_keys):
    for scores in (first, second):
        if not isinstance(scores, list) or len(scores) != len(rubric_keys):
            raise ValueError("Expected the existing eight-score arrays for both human raters")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or not 1 <= value <= 5 or int(value) != value for value in scores):
            raise ValueError("Human rater scores must be integers on the existing 1–5 scale")
    return tuple(first), tuple(second)


@dataclass(frozen=True)
class Candidate:
    essay: Essay
    provenance: dict

    @property
    def total(self):
        return sum(self.essay.original_scores.values())


def load_heldout_candidates(source, rubric_keys, *, min_paras, max_paras):
    """Cross-check test membership, raw text, human scores, and training overlap."""
    paths = {key: Path(value) for key, value in source.model_dump().items()}
    hashes = {key: file_hash(path) for key, path in paths.items() if key != "raw_test_dir"}
    audit = json.loads(paths["audit_manifest"].read_text(encoding="utf-8"))
    for split in ("train", "test"):
        if hashes[f"{split}_jsonl"] != audit[split]["sha256"]:
            raise ValueError(f"The {split} source differs from the existing scorer-data audit")

    training_exact, training_compact = set(), set()
    training_count = 0
    for _, row in iter_source(paths["train_jsonl"]):
        _, text = source_question_essay(row)
        training_exact.add(text_hash(text))
        training_compact.add(text_hash(text, ignore_whitespace=True))
        training_count += 1
    if training_count != audit["train"]["rows"] or training_exact != set(audit["train"]["essay_hashes"]):
        raise ValueError("Training text extraction disagrees with the existing audit")

    # Use the raw source filename as the stable essay ID. Scores disambiguate duplicate texts.
    raw_index = defaultdict(list)
    for path in sorted(paths["raw_test_dir"].glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        question, text = row["essay_question"]["prompt"], row["essay_answer"]["text"]
        scores = row["score"]["personal"]["analytic"]
        raters = checked_raters([scores[key]["score"][0] for key in rubric_keys],
                                [scores[key]["score"][1] for key in rubric_keys], rubric_keys)
        raw_index[(question, text, *raters)].append(path)

    candidates, excluded, seen = [], [], set()
    distribution, overlap_count, test_count = Counter(), 0, 0
    for line_number, row in iter_source(paths["test_jsonl"]):
        test_count += 1
        question, text = source_question_essay(row)
        paragraphs = original_paragraphs(text)
        distribution[len(paragraphs)] += 1
        compact = text_hash(text, ignore_whitespace=True)
        exact = text_hash(text)
        overlap = compact in training_compact
        overlap_count += int(overlap)
        reason = None
        if overlap:
            reason = "scorer_training_overlap"
        elif len(paragraphs) < 2:
            reason = "no_original_paragraph_boundary"
        elif not min_paras <= len(paragraphs) <= max_paras:
            reason = "paragraph_count_outside_bounds"
        else:
            raters = checked_raters(row["grader_1_scores"], row["grader_2_scores"], rubric_keys)
            raw_paths = raw_index.get((question, text, *raters), [])
            if len(raw_paths) != 1:
                reason = "missing_or_ambiguous_raw_source"
            elif compact in seen:
                reason = "duplicate_test_essay"
        if reason:
            excluded.append({"source_line": line_number, "reason": reason, "paragraph_count": len(paragraphs)})
            continue
        raw_path = raw_paths[0]
        # Verified in essay_scoring_llm.scaling.label_from_raters: rater1 + rater2 - 1.
        original_scores = {key: float(a + b - 1) for key, a, b in zip(rubric_keys, *raters)}
        essay = Essay(essay_id=raw_path.stem, prompt=question, paragraphs=paragraphs,
                      original_scores=original_scores)
        provenance = {"essay_id": essay.essay_id, "source_line": line_number,
                      "raw_source": str(raw_path), "raw_source_sha256": file_hash(raw_path),
                      "original_text_sha256": exact, "whitespace_ignored_sha256": compact,
                      "rater_1": list(raters[0]), "rater_2": list(raters[1]),
                      "original_total": sum(original_scores.values()), "paragraph_count": len(paragraphs)}
        candidates.append(Candidate(essay, provenance))
        seen.add(compact)
    if test_count != audit["test"]["rows"]:
        raise ValueError("Test row count differs from the existing scorer-data audit")
    # Detect input changes during selection rather than mixing versions in one split.
    if any(file_hash(paths[key]) != digest for key, digest in hashes.items()):
        raise ValueError("A source file changed while preparing the split")
    report = {"source_paths": {key: str(path.resolve()) for key, path in paths.items()},
              "source_sha256": hashes, "train_rows": training_count, "test_rows": test_count,
              "original_paragraph_distribution": dict(sorted(distribution.items())),
              "training_overlap_rows": overlap_count, "eligible_count": len(candidates),
              "excluded_counts": dict(Counter(row["reason"] for row in excluded)),
              "paragraph_policy": "Original CRLF/CR/LF boundaries; ignore blank lines, preserve paragraph contents",
              "original_scores_source": "Human rater_1 + rater_2 - 1, each rubric on 1–9 scale; used for stratification",
              "original_scores_are_kanana_predictions": False,
              "leakage_check": "Exact and whitespace-ignored essay text against the scorer training split"}
    return candidates, report, excluded


def stratified_sample(candidates, *, count, seed):
    if count < 3 or len(candidates) < count:
        raise ValueError("Need at least three requested essays and enough eligible candidates")
    lower, upper = quantiles([row.total for row in candidates], n=3, method="inclusive")
    names = ("low", "middle", "high")
    groups = {name: [] for name in names}
    for row in sorted(candidates, key=lambda item: item.essay.essay_id):
        name = "low" if row.total <= lower else "middle" if row.total <= upper else "high"
        groups[name].append(row)
    rng = random.Random(seed)
    quotas = {name: count // 3 + int(index < count % 3) for index, name in enumerate(names)}
    draws = {}
    for name in names:
        if len(groups[name]) < quotas[name]:
            raise ValueError(f"Insufficient {name} stratum: {len(groups[name])} available, {quotas[name]} required")
        draws[name] = rng.sample(groups[name], quotas[name])
    # Interleave strata; the first five of a ten-essay pilot remain 2/2/1.
    selected = [(name, draws[name][index]) for index in range(max(quotas.values()))
                for name in names if index < quotas[name]]
    return [row for _, row in selected], {
        "seed": seed, "score": "sum of the eight original human-derived rubric scores",
        "quantile_method": "inclusive (linear interpolation), equal scores stay in the same stratum",
        "cutpoints": [lower, upper], "intervals": {"low": "score <= q1", "middle": "q1 < score <= q2", "high": "q2 < score"},
        "available": {name: len(groups[name]) for name in names}, "quotas": quotas,
        "selected": [{**row.provenance, "stratum": name} for name, row in selected]}

