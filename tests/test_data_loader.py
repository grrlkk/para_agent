from collections import Counter
import json
from pathlib import Path

import pytest

from para_agent.artifacts import file_hash
from para_agent.config import HeldOutSourceConfig
from para_agent.data.loader import (Candidate, Essay, checked_raters, load_essays,
                                    load_heldout_candidates, original_paragraphs,
                                    source_question_essay, stratified_sample, text_hash)


def corpus(tmp_path, keys, test_texts, train_texts=()):
    raw_dir = tmp_path / "raw_test"
    raw_dir.mkdir()
    def source_row(text):
        return {"user": f"질문: 문항\n에세이: {text}\n핵심 키워드: 본문에 포함하면 안 됨",
                "grader_1_scores": [3.0] * len(keys), "grader_2_scores": [4.0] * len(keys)}
    paths = {}
    for split, texts in (("train", train_texts), ("test", test_texts)):
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(source_row(text), ensure_ascii=False) + "\n" for text in texts))
        paths[split] = path
    for index, text in enumerate(test_texts):
        raw = {"essay_question": {"prompt": "문항"}, "essay_answer": {"text": text},
               "score": {"personal": {"analytic": {key: {"score": [3, 4]} for key in keys}}}}
        (raw_dir / f"essay_{index}.json").write_text(json.dumps(raw, ensure_ascii=False))
    audit = {split: {"sha256": file_hash(paths[split]), "rows": len(texts)}
             for split, texts in (("train", train_texts), ("test", test_texts))}
    audit["train"]["essay_hashes"] = sorted({text_hash(text) for text in train_texts})
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    evidence = tmp_path / "evidence.txt"
    evidence.write_text("Synthetic audit fixture, not a real training claim.")
    return HeldOutSourceConfig(test_jsonl=str(paths["test"]), train_jsonl=str(paths["train"]),
                               raw_test_dir=str(raw_dir), audit_manifest=str(audit_path), training_evidence=str(evidence))


def test_original_boundaries_preserve_characters_and_do_not_invent_paragraphs():
    text = " 첫 문단. 문장이 두 개여도 한 문단.  \r\n\r\n두 번째 문단.\n \n 마지막 문단. "
    assert original_paragraphs(text) == [" 첫 문단. 문장이 두 개여도 한 문단.  ", "두 번째 문단.", " 마지막 문단. "]
    assert original_paragraphs("첫 문장. 둘째 문장. 셋째 문장.") == ["첫 문장. 둘째 문장. 셋째 문장."]


def test_source_parser_removes_only_prompt_metadata_and_preserves_essay_edges():
    text = "  원문의 앞뒤 공백. \n둘째 문단.  "
    question, parsed = source_question_essay({"user": f"질문: 문항\n에세이: {text}\n핵심 키워드: 단어"})
    assert question == "문항" and parsed == text
    with pytest.raises(ValueError, match="existing"):
        source_question_essay({"text": text})


def test_heldout_filter_excludes_training_duplicates_even_with_different_whitespace(tmp_path, contracts):
    kept = " 첫째 문단. \r\n둘째 문단.\r\n\r\n셋째 문단. "
    leaked = "학습 원문.\n다음 문단.\n끝 문단."
    duplicate = "첫째문단.\n둘째문단.\n셋째문단."
    source = corpus(tmp_path, contracts["rubric_keys"],
                    [kept, leaked, "한 문단. 두 문장. 세 문장.", "두 문단\n뿐인 글", duplicate],
                    ["학습원문. 다음문단. 끝문단."])
    candidates, audit, excluded = load_heldout_candidates(source, contracts["rubric_keys"], min_paras=3, max_paras=6)
    assert len(candidates) == 1
    assert candidates[0].essay.paragraphs == original_paragraphs(kept)
    assert candidates[0].essay.original_scores == dict.fromkeys(contracts["rubric_keys"], 6.0)
    assert candidates[0].provenance["source_line"] == 1
    assert candidates[0].provenance["original_text_sha256"] == text_hash(kept)
    assert audit["training_overlap_rows"] == 1
    assert audit["excluded_counts"] == {"scorer_training_overlap": 1, "no_original_paragraph_boundary": 1,
                                        "paragraph_count_outside_bounds": 1, "duplicate_test_essay": 1}
    assert len(excluded) == 4
    assert audit["original_scores_are_kanana_predictions"] is False


@pytest.mark.parametrize("split", ["test", "train"])
def test_source_must_match_existing_audit(tmp_path, contracts, split):
    source = corpus(tmp_path, contracts["rubric_keys"], ["첫째\n둘째\n셋째"])
    path = Path(getattr(source, f"{split}_jsonl"))
    with path.open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match=f"{split} source differs"):
        load_heldout_candidates(source, contracts["rubric_keys"], min_paras=3, max_paras=6)


def test_raw_source_score_mismatch_is_excluded_not_guessed(tmp_path, contracts):
    source = corpus(tmp_path, contracts["rubric_keys"], ["첫째\n둘째\n셋째"])
    raw_path = Path(source.raw_test_dir) / "essay_0.json"
    raw = json.loads(raw_path.read_text())
    raw["score"]["personal"]["analytic"][contracts["rubric_keys"][0]]["score"][0] = 2
    raw_path.write_text(json.dumps(raw))
    candidates, audit, _ = load_heldout_candidates(source, contracts["rubric_keys"], min_paras=3, max_paras=6)
    assert candidates == []
    assert audit["excluded_counts"] == {"missing_or_ambiguous_raw_source": 1}


@pytest.mark.parametrize("value", [0, 6, 2.5, float("nan"), True])
def test_human_scale_is_validated_before_score_conversion(contracts, value):
    with pytest.raises(ValueError, match="1–5 scale"):
        checked_raters([value] + [3] * 7, [4] * 8, contracts["rubric_keys"])


def candidates_for_scores(contracts, values):
    return [Candidate(Essay(essay_id=f"sample_{index:03}", prompt="문항", paragraphs=["첫째", "둘째", "셋째"],
                            original_scores=dict.fromkeys(contracts["rubric_keys"], float(value))),
                       {"essay_id": f"sample_{index:03}"}) for index, value in enumerate(values)]


def test_stratification_is_seeded_tie_preserving_and_reproducible(contracts):
    candidates = candidates_for_scores(contracts, [1] * 10 + [5] * 10 + [9] * 10)
    first, manifest = stratified_sample(candidates, count=10, seed=42)
    second, repeated = stratified_sample(list(reversed(candidates)), count=10, seed=42)
    assert first == second and manifest == repeated
    assert len({item.essay.essay_id for item in first}) == 10
    assert manifest["available"] == {"low": 10, "middle": 10, "high": 10}
    assert manifest["quotas"] == {"low": 4, "middle": 3, "high": 3}
    assert Counter(row["stratum"] for row in manifest["selected"][:5]) == {"low": 2, "middle": 2, "high": 1}
    third, _ = stratified_sample(candidates, count=10, seed=43)
    assert first != third


def test_stratification_does_not_duplicate_rows_or_move_ties_to_fill_strata(contracts):
    with pytest.raises(ValueError, match="enough eligible"):
        stratified_sample(candidates_for_scores(contracts, [1, 2, 3]), count=10, seed=42)
    with pytest.raises(ValueError, match="Insufficient middle"):
        stratified_sample(candidates_for_scores(contracts, [5] * 15), count=10, seed=42)


def test_saved_split_must_have_unique_ids_and_original_paragraphs(tmp_path, contracts):
    path = tmp_path / "split.jsonl"
    row = candidates_for_scores(contracts, [4])[0].essay
    path.write_text(row.model_dump_json() + "\n" + row.model_dump_json() + "\n")
    with pytest.raises(ValueError, match="unique essay IDs"):
        load_essays(path, min_paras=3, max_paras=6)
