import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from para_agent.config import PROJECT_ROOT


def cli():
    spec = importlib.util.spec_from_file_location("run_experiment", PROJECT_ROOT / "scripts/run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("diag_unit", ["both", "para", "cum"])
def test_mock_end_to_end_and_cache(tmp_path, capsys, contracts, cfg, diag_unit):
    main = cli()
    cfg.diag_unit = diag_unit
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump()))
    args = ["--mock", "--config", str(config_path), "--output-dir", str(tmp_path / "runs"),
            "--cache-dir", str(tmp_path / "cache")]
    assert main(args) == 0
    first_stdout = json.loads(capsys.readouterr().out)
    first = Path(first_stdout["run_dir"])
    assert first_stdout["real_api_calls"] == 0
    assert first_stdout["mock_transport_calls"] == 25
    snapshot = yaml.safe_load((first / "config.yaml").read_text())
    assert snapshot["provenance"]["mock"] is True
    assert len(snapshot["provenance"]["prompt_hashes"]) == 5
    originals = [json.loads(line) for line in (PROJECT_ROOT / "tests/fixtures/mock_essays.jsonl").read_text().splitlines()]
    for essay in originals:
        paragraphs = essay["paragraphs"]
        for condition in ("whole_once", "incremental"):
            directory = first / condition / essay["essay_id"]
            final = read(directory / "final.json")
            assert final["Y"] == paragraphs
            assert final["final_text"] == "\n\n".join(paragraphs)
            assert final["profile"]["prompt"] == essay["prompt"]
            assert final["profile"]["register"] == '해라체'
            turns = sorted(directory.glob("turn_*.json"))
            assert len(turns) == (1 if condition == "whole_once" else len(paragraphs))
            for path in turns:
                turn = read(path)
                diagnoses = turn["diag"] if condition == "incremental" and diag_unit == "both" else {turn["diag"]["unit"]: turn["diag"]}
                assert set(diagnoses) == ({"para", "cum"} if condition == "incremental" and diag_unit == "both"
                                          else {diag_unit if condition == "incremental" else "cum"})
                for unit, diagnosis in diagnoses.items():
                    assert diagnosis["unit"] == unit
                    assert set(diagnosis["rubric_scores"]) == set(contracts["rubric_keys"])
                    assert set(diagnosis["features"]) == set(contracts["feature_keys"])
                    assert all(value is None for value in diagnosis["features"].values())
                assert turn["issue_limit"] == 4 * (len(paragraphs) if condition == "whole_once" else 1)
                assert turn["kb_entry_ids"]
            if condition == "whole_once":
                assert final["memory"]["turn_diags"] == []
                assert read(turns[0])["memory_update"] is None
            else:
                assert len(final["memory"]["turn_diags"]) == len(paragraphs)
                assert len(final["memory"]["turn_summaries"]) == len(paragraphs)
                for i, path in enumerate(turns):
                    turn = read(path)
                    assert turn["prefix"] == paragraphs[:i]
                    assert len(turn["state_before"]["memory"]["turn_diags"]) == i
                    views = turn["evidence"].values() if diag_unit == "both" else [turn["evidence"]]
                    for view in views:
                        assert set(view["rubric_deltas"].values()) == ({None} if i == 0 else {0.0})
                    assert final["memory"]["turn_diags"][i] == {"turn": i, **turn["diag"]}
    calls = [json.loads(line) for line in (first / "llm_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 25
    assert Counter(row["agent"] for row in calls) == {"summarizer": 9, "reviser": 9, "memory_update": 7}
    assert all(row["tokens"]["total_tokens"] == 0 and row["mock"] for row in calls)
    assert all(row["status"] == "ok" and row["raw_response"] and row["prompt_sha256"] for row in calls)
    for call in calls:
        if diag_unit == "both" and call["agent"] == "summarizer" and call["condition"] == "incremental":
            payload = json.loads(call["messages"][1]["content"])
            assert set(payload["evidence"]) == {"para", "cum"}
    before = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    assert main(args) == 0
    second_stdout = json.loads(capsys.readouterr().out)
    assert second_stdout["mock_transport_calls"] == 0
    second = Path(second_stdout["run_dir"])
    assert second != first
    assert all(path.read_bytes() == contents for path, contents in before.items())
    second_calls = [json.loads(line) for line in (second / "llm_calls.jsonl").read_text().splitlines()]
    assert len(second_calls) == 25 and all(row["cache_hit"] for row in second_calls)


@pytest.mark.parametrize("diag_unit", ["para", "both"])
def test_memory_off_and_prev_plus_x(tmp_path, cfg, capsys, diag_unit):
    config = cfg.model_dump()
    config.update(use_memory=False, reviser_input="prev_plus_x", diag_unit=diag_unit, delta_mode="cum")
    config_path = tmp_path / "off.yaml"
    config_path.write_text(yaml.safe_dump(config))
    cli()(["--mock", "--config", str(config_path), "--condition", "incremental",
           "--output-dir", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache")])
    run = Path(json.loads(capsys.readouterr().out)["run_dir"])
    assert not (run / "whole_once").exists()
    for turn_path in run.glob("incremental/*/turn_*.json"):
        turn = read(turn_path)
        if diag_unit == "both":
            assert set(turn["diag"]) == {"para", "cum"}
        else:
            assert turn["diag"]["unit"] == "para"
        views = turn["evidence"].values() if diag_unit == "both" else [turn["evidence"]]
        for view in views:
            assert all(value is None for value in view["rubric_deltas"].values())
        for state in ("state_before", "state_after"):
            assert turn[state]["memory"] == {"turn_diags": [], "error_notes": [], "turn_summaries": []}
            assert turn[state]["profile"]["style_notes"] is None
        assert not turn["memory_update_applied"]
    calls = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text().splitlines()]
    for call in calls:
        if call["agent"] == "reviser":
            payload = json.loads(call["messages"][1]["content"])
            assert isinstance(payload["rev_in"], dict)
            assert (payload["rev_in"]["prev"] is None) == (call["turn"] == 0)
            assert (payload["prev_tail"] is None) == (call["turn"] == 0)


def test_live_run_fails_before_creating_artifacts(tmp_path):
    with pytest.raises(SystemExit) as error:
        cli()(["--output-dir", str(tmp_path / "runs")])
    assert error.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_nonmock_provider_rejected(tmp_path, cfg):
    cfg.llm.agent.provider = "real-provider"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()))
    with pytest.raises(SystemExit):
        cli()(["--mock", "--config", str(path), "--output-dir", str(tmp_path / "runs")])
    assert not (tmp_path / "runs").exists()

