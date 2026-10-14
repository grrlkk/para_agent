import ast
import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from para_agent.config import PROJECT_ROOT
from para_agent.data.loader import Essay
from para_agent.diagnose.mock import MockDiagnoser
from para_agent.memory.schema import MemoryUpdate, Profile, SessionMemory


def test_kb_immutable_and_explicit_aliases(contracts, kb):
    path = PROJECT_ROOT / "data/kb/features.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "7338b9216b8dd3246d6491abd343cc264556bd51474755cfe4c28040031622cc"
    original = PROJECT_ROOT.parent / "imple/features.jsonl"
    if original.is_file():
        assert path.read_bytes() == original.read_bytes()
    assert len(kb.entries) == len(kb.key_map) == 29
    assert {entry["code"] for entry in kb.entries} == set(kb.key_map)
    assert set(kb.key_map.values()) == set(contracts["feature_keys"])
    assert sum(code != key for code, key in kb.key_map.items()) == 25
    rubric_map = yaml.safe_load((PROJECT_ROOT / "configs/rubric_diag_map.yaml").read_text())
    assert set(rubric_map) == set(contracts["rubric_keys"])
    assert set(item for items in rubric_map.values() for item in items) <= {row["diagnostic_item"] for row in kb.entries}


def test_contract_matches_legacy_source_without_importing_models(contracts):
    path = Path(contracts["source"])
    if not path.exists():
        pytest.skip("Optional local legacy checkout is absent")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == contracts["source_sha256"]
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"RUBRIC_KEYS", "RUBRIC_NAMES_KO", "FEAK_NAMES"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    assert values["RUBRIC_KEYS"] == contracts["rubric_keys"]
    assert values["RUBRIC_NAMES_KO"] == contracts["rubric_names_ko"]
    assert values["FEAK_NAMES"] == contracts["feature_keys"]


def test_mock_diagnosis_is_deterministic_and_uses_revised_prefix(contracts):
    diagnoser = MockDiagnoser(contracts["rubric_keys"], contracts["feature_keys"])
    captured = []
    original = diagnoser.scorer.score
    def score(text, prompt):
        captured.append(text)
        return original(text, prompt)
    diagnoser.scorer.score = score
    first = diagnoser.diagnose("현재 원문", ["앞 수정본"], "cum", "문항")
    assert first == diagnoser.diagnose("현재 원문", ["앞 수정본"], "cum", "문항")
    diagnoser.diagnose("현재 원문", ["앞 수정본"], "para", "문항")
    assert captured == ["앞 수정본\n\n현재 원문", "앞 수정본\n\n현재 원문", "현재 원문"]


def test_profile_patch_preserves_unset_fields_and_prompt():
    profile = Profile(prompt="문항", topic="주제", register="해라체")
    update = MemoryUpdate.model_validate({"error_notes": [], "profile_update": {"style_notes": "짧은 문장"}, "turn_summary": "요약"})
    profile.apply(update)
    assert profile.model_dump() == {"prompt": "문항", "topic": "주제", "register": "해라체", "style_notes": "짧은 문장", "thesis": None}
    assert "register" in Profile.model_json_schema()["properties"]
    assert "register_" not in Profile.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        MemoryUpdate.model_validate({"error_notes": [], "profile_update": {"prompt": "다른 문항"}, "turn_summary": "요약"})
    left, right = SessionMemory(), SessionMemory()
    left.apply(update)
    assert right.turn_summaries == []


@pytest.mark.parametrize("essay_id", ["../escape", "/absolute", "a/b"])
def test_essay_id_cannot_escape_run_directory(essay_id):
    with pytest.raises(ValidationError):
        Essay(essay_id=essay_id, prompt="문항", paragraphs=["문단"], original_scores={})

