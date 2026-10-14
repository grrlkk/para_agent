from collections import Counter
import copy
import json

import pytest

from para_agent.artifacts import file_hash
from para_agent.config import project_path
from para_agent.diagnose.scorer_adapter import load_scorer_settings
from para_agent.diagnose.scorer_metadata import merge_heading_evidence, validate_metadata_override


def test_adding_a_genre_preserves_all_existing_entries():
    headings = tuple(f"rubric_{index}" for index in range(8))
    base = {"allowed_labels": ["설명"], "questions": {"existing": {"genre": "설명", "method": "original", "rows": 9}}}
    before = copy.deepcopy(base)
    combined, added = merge_heading_evidence(base, {"설명": set(headings)},
                                             {"existing": Counter({headings: 3}), "new": Counter({headings: 4})})
    assert base == before
    assert combined["questions"]["existing"] == base["questions"]["existing"]
    assert set(added) == {"new"}
    assert added["new"] == {"genre": "설명", "method": "offline_feedback_heading_set", "rows": 4, "distinct_heading_sets": 1}


def test_ambiguous_unknown_or_conflicting_genres_are_not_guessed():
    headings = tuple(f"rubric_{index}" for index in range(8))
    base = {"questions": {"existing": {"genre": "정서"}}}
    rule = {"설명": set(headings)}
    with pytest.raises(ValueError, match="Conflicting rubric"):
        merge_heading_evidence(base, rule, {"new": Counter({headings: 1, ("unrecognized",): 1})})
    with pytest.raises(ValueError, match="cannot identify"):
        merge_heading_evidence(base, rule, {"new": Counter({("unrecognized",): 1})})
    with pytest.raises(ValueError, match="conflicts with existing"):
        merge_heading_evidence(base, rule, {"existing": Counter({headings: 1})})


def test_supplement_refuses_changed_base_or_altered_existing_entries(tmp_path):
    base = {"allowed_labels": ["설명", "정서"], "questions": {"old": {"genre": "설명"}}}
    base_path, override_path = tmp_path / "base.json", tmp_path / "genres.json"
    base_path.write_text(json.dumps(base))
    override = {**copy.deepcopy(base), "provenance": {"base_metadata_sha256": file_hash(base_path)}}
    override["questions"]["new"] = {"genre": "정서"}
    override_path.write_text(json.dumps(override))
    validate_metadata_override(base_path, override_path)
    override["questions"]["old"]["genre"] = "정서"
    override_path.write_text(json.dumps(override))
    with pytest.raises(ValueError, match="every existing question"):
        validate_metadata_override(base_path, override_path)
    base_path.write_text(json.dumps(base) + "\n")
    with pytest.raises(ValueError, match="different original metadata"):
        validate_metadata_override(base_path, override_path)


def test_metadata_override_changes_only_the_metadata_directory(cfg):
    configured = load_scorer_settings(cfg.scorer)
    original = load_scorer_settings(cfg.scorer.model_copy(update={"metadata_dir": None}))
    assert configured["paths"]["metadata"] == project_path(cfg.scorer.metadata_dir)
    configured["paths"]["metadata"] = original["paths"]["metadata"]
    assert configured == original
