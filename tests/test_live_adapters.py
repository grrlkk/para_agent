import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from para_agent.config import PROJECT_ROOT, project_path
from para_agent.diagnose.feature_adapter import FeatureAdapter
from para_agent.diagnose.scorer_adapter import KananaScorerAdapter, load_scorer_settings
from para_agent.diagnose.service import Diagnoser, build_diagnoser
from para_agent.kb.validate import validate_kb


class NativeScorerStub:
    fingerprint = "test-fingerprint"

    def __init__(self, expected=None, error=None):
        self.expected = expected if expected is not None else [i + 1.125 for i in range(8)]
        self.error = error
        self.calls = []

    def score(self, question, text, *, use_cache):
        self.calls.append((question, text, use_cache))
        if self.error:
            raise self.error
        return SimpleNamespace(to_dict=lambda: {"expected": self.expected, "cache_hit": use_cache,
                                                "input_tokens": 123})


def test_native_score_order_and_settings_preserved(cfg, contracts, tmp_path):
    backend = NativeScorerStub()
    path = tmp_path / "calls.jsonl"
    scorer = KananaScorerAdapter(cfg.scorer, contracts["rubric_keys"], log_path=path, backend=backend)
    scores, reasons = scorer.score("학습자 문단", "작문 문항", use_cache=False)
    assert list(scores.items()) == list(zip(contracts["rubric_keys"], backend.expected))
    assert reasons is None
    assert backend.calls == [("작문 문항", "학습자 문단", False)]
    record = json.loads(path.read_text())
    assert record["result"]["expected"] == backend.expected
    assert record["model_fingerprint"] == backend.fingerprint
    source = yaml.safe_load(Path(cfg.scorer.source_config).read_text(encoding="utf-8"))
    settings = load_scorer_settings(cfg.scorer)
    assert settings["scorer"] == source["scorer"]
    assert str(settings["paths"]["policy_base"]) == source["paths"]["policy_base"]
    assert str(settings["paths"]["scorer_adapter"]) == source["paths"]["scorer_adapter"]
    assert settings["paths"]["output"] == project_path(cfg.scorer.cache_dir)


@pytest.mark.parametrize("values", [[1.0] * 7, [float("nan")] + [1.0] * 7])
def test_invalid_native_scores_fail_without_substitution(cfg, contracts, tmp_path, values):
    path = tmp_path / "calls.jsonl"
    scorer = KananaScorerAdapter(cfg.scorer, contracts["rubric_keys"], log_path=path,
                                backend=NativeScorerStub(expected=values))
    with pytest.raises(ValueError, match="invalid expected scores"):
        scorer.score("문단", "문항")
    assert json.loads(path.read_text())["status"] == "error"


def test_input_overflow_is_recorded_and_never_silently_truncated(cfg, contracts, tmp_path):
    error = ValueError("too long")
    error.tokens, error.limit = 3073, 3072
    path = tmp_path / "calls.jsonl"
    scorer = KananaScorerAdapter(cfg.scorer, contracts["rubric_keys"], log_path=path,
                                backend=NativeScorerStub(error=error))
    with pytest.raises(ValueError, match="too long"):
        scorer.score("긴 글", "문항")
    record = json.loads(path.read_text())
    assert record["status"] == "input_too_long"
    assert record["input_tokens"] == 3073 and record["max_input_tokens"] == 3072
    assert record["truncated"] is False


@pytest.mark.parametrize("unit,count", [("para", 1), ("cum", 1), ("cum", 2)])
def test_scope_and_unavailable_values_do_not_leak_legacy_imputations(cfg, contracts, kb, tmp_path, unit, count):
    details = {key: {"value": 12.5, "imputed": False, "source": "test"}
               for key in contracts["feature_keys"]}
    both_keys = [kb.key_map[row["code"]] for row in kb.entries if row["unit_scope"] == "both"]
    details[both_keys[0]]["imputed"] = True
    details[both_keys[1]]["value"] = float("nan")
    details[both_keys[2]]["value"] = None
    worker = lambda text: {"details": details, "device": "test"}
    path = tmp_path / "calls.jsonl"
    adapter = FeatureAdapter(cfg.features, contracts["feature_keys"], kb, log_path=path, worker=worker)
    result = adapter.extract("검증 글", unit=unit, paragraph_count=count)
    assert list(result) == contracts["feature_keys"]
    assert all(result[key] is None for key in both_keys[:3])
    cumulative = [kb.key_map[row["code"]] for row in kb.entries if row["unit_scope"] == "cum"]
    assert len(cumulative) == 8
    assert all(result[key] == (12.5 if unit == "cum" and count >= 2 else None) for key in cumulative)
    assert json.loads(path.read_text())["unavailable"][both_keys[0]] == "legacy_imputed_value"


def test_changed_feature_output_keys_fail_instead_of_being_guessed(cfg, contracts, kb, tmp_path):
    adapter = FeatureAdapter(cfg.features, contracts["feature_keys"], kb,
                             log_path=tmp_path / "calls.jsonl", worker=lambda text: {"details": {"unknown": 1}})
    with pytest.raises(ValueError, match="different feature keys"):
        adapter.extract("문단", unit="para", paragraph_count=1)


def test_both_diagnoses_keep_distinct_text_and_raw_outputs(contracts):
    calls = []
    def score(text, prompt, *, use_cache):
        calls.append((text, prompt, use_cache))
        return dict.fromkeys(contracts["rubric_keys"], float(len(text))), None
    def extract(text, *, unit, paragraph_count):
        return dict.fromkeys(contracts["feature_keys"], float(paragraph_count))
    diagnoser = Diagnoser(SimpleNamespace(score=score), SimpleNamespace(extract=extract))
    pair = diagnoser.diagnose("이번 원문", ["앞 수정본"], "both", "문항", use_cache=False)
    assert calls == [("이번 원문", "문항", False), ("앞 수정본\n\n이번 원문", "문항", False)]
    assert set(pair.para.features.values()) == {1.0}
    assert set(pair.cum.features.values()) == {2.0}
    assert pair.para.rubric_scores != pair.cum.rubric_scores


def test_kb_validator_reports_original_aliases_and_rejects_wrong_mapping(cfg, kb, tmp_path):
    path = PROJECT_ROOT / "data/kb/features.jsonl"
    mapping = PROJECT_ROOT / "configs/feature_key_map.yaml"
    result = validate_kb(path, mapping, cfg.features.package_path)
    assert result["ok"] and result["direct_code_match_count"] == 4
    assert result["mapped_code_match_count"] == 29 and len(result["aliases"]) == 25
    wrong = dict(kb.key_map)
    wrong[next(iter(wrong))] = "not_a_real_feature"
    bad_map = tmp_path / "mapping.yaml"
    bad_map.write_text(yaml.safe_dump(wrong))
    result = validate_kb(path, bad_map, cfg.features.package_path)
    assert not result["ok"]
    assert any("every actual feature key" in error for error in result["errors"])


def test_live_factory_refuses_a_stale_recorded_contract(cfg, contracts, tmp_path):
    contracts["rubric_keys"][0] = "unconfirmed_rubric"
    path = tmp_path / "contracts.yaml"
    path.write_text(yaml.safe_dump(contracts))
    cfg.paths.contracts = str(path)
    with pytest.raises(ValueError, match="confirm before proceeding"):
        build_diagnoser(cfg, log_path=tmp_path / "calls.jsonl")
