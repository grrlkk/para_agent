import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from para_agent.artifacts import file_hash, write_json
from para_agent.config import PROJECT_ROOT
from para_agent.data.pilot import load_stage3_pilot
from para_agent.diagnose.mock import MockDiagnoser
from para_agent.llm.client import MockTransport


def pilot_fixture(tmp_path, cfg):
    cfg.data.n_essays = 5
    rows, selected = [], []
    for index in range(10):
        essay_id = f"pilot-{index}"
        paragraphs = [f"{index} 원문 {turn}." for turn in range(3 + index % 4)]
        row = {"essay_id": essay_id, "prompt": "문항", "paragraphs": paragraphs, "original_scores": {}}
        rows.append(row)
        raw = tmp_path / f"{essay_id}.json"
        write_json(raw, {"essay_question": {"prompt": "문항"}, "essay_answer": {"text": "\r\n".join(paragraphs)}})
        selected.append({"essay_id": essay_id, "raw_source": str(raw), "raw_source_sha256": file_hash(raw)})
    split = tmp_path / "split.jsonl"
    split.write_text("\n".join(json.dumps(row) for row in rows))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"stage": 2, "ok": True, "essay_count": 10, "split": str(split),
                          "split_sha256": file_hash(split), "sampling": {"selected": selected},
                          "audit": {"source_sha256": {}, "source_paths": {}}})
    return manifest, rows


def test_stage3_uses_first_five_and_preserves_original_boundaries(tmp_path, cfg):
    manifest, rows = pilot_fixture(tmp_path, cfg)
    essays, provenance = load_stage3_pilot(manifest, cfg)
    assert [e.model_dump() for e in essays] == rows[:5]
    assert provenance["essay_ids"] == [r["essay_id"] for r in rows[:5]]
    cfg.data.n_essays = 6
    with pytest.raises(ValueError, match="exactly five"):
        load_stage3_pilot(manifest, cfg)


@pytest.mark.parametrize("change", ["split", "raw", "order"])
def test_stage3_rejects_changed_data(tmp_path, cfg, change):
    manifest, rows = pilot_fixture(tmp_path, cfg)
    info = json.loads(manifest.read_text())
    if change == "split":
        Path(info["split"]).write_text("changed")
    elif change == "raw":
        Path(info["sampling"]["selected"][0]["raw_source"]).write_text("changed")
    else:
        info["sampling"]["selected"].reverse()
        manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError):
        load_stage3_pilot(manifest, cfg)


@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
def test_stage3_live_branch_is_isolated_to_five_essays(tmp_path, cfg, monkeypatch, contracts, capsys, provider_name):
    manifest, rows = pilot_fixture(tmp_path, cfg)
    cfg.llm.agent.provider = provider_name
    cfg.llm.agent.model = "gpt-6-luna" if provider_name == "openai" else "claude-haiku-5-5"
    cfg.llm.agent.temperature = 0 if provider_name == "openai" else None
    cfg.llm.agent.reasoning_effort = "none"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump()))

    class FakeLive(MockTransport):
        provider = provider_name

        def __init__(self, *args, **kwargs):
            super().__init__(PROJECT_ROOT / "tests/fixtures/mock_responses.json")

    transport_name = ("openai_transport.OpenAITransport" if provider_name == "openai" else
                      "anthropic_transport.AnthropicTransport")
    monkeypatch.setattr(f"para_agent.llm.{transport_name}", FakeLive)
    monkeypatch.setattr("para_agent.diagnose.service.build_diagnoser", lambda *a, **kw:
                        MockDiagnoser(contracts["rubric_keys"], contracts["feature_keys"]))
    spec = importlib.util.spec_from_file_location("run_experiment", PROJECT_ROOT / "scripts/run_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--stage", "3", "--config", str(config_path), "--split-manifest", str(manifest),
                        "--output-dir", str(tmp_path / "runs"), "--cache-dir", str(tmp_path / "cache")]) == 0
    report = json.loads(capsys.readouterr().out)
    run = Path(report["run_dir"])
    expected = 10 + 3 * sum(len(row["paragraphs"]) for row in rows[:5])
    assert report["real_api_calls"] == report["expected_logical_calls"] == expected
    assert report["ok"] and report["stage"] == 3 and report["final_texts"] == 10
    logs = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text().splitlines()]
    assert {row["essay_id"] for row in logs} == {row["essay_id"] for row in rows[:5]}
    assert all(not row["mock"] for row in logs)
    assert {row["agent"] for row in logs} == {"summarizer", "reviser", "memory_update"}
    assert all("original_scores" not in row["messages"][1]["content"] for row in logs)
    assert all(json.loads(path.read_text())["mock"] is False for path in run.glob("*/*/final.json"))
