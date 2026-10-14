import json

import pytest

from para_agent.agents.summarizer import Brief, summarize
from para_agent.config import PROJECT_ROOT
from para_agent.diagnose.schema import Evidence
from para_agent.llm.client import Client, MockTransport
from para_agent.memory.schema import Profile


def make_client(tmp_path, cfg, transport):
    return Client(model=cfg.llm.agent, transport=transport, prompts=PROJECT_ROOT / "prompts",
                  cache=tmp_path / "cache", call_log=tmp_path / "calls.jsonl")


class ScriptedTransport(MockTransport):
    def __init__(self, responses):
        super().__init__(PROJECT_ROOT / "tests/fixtures/mock_responses.json")
        self.script = iter(responses)
        self.seen = []

    def complete(self, *, agent, messages, params):
        self.call_count += 1
        self.seen.append(messages[:])
        return next(self.script), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def test_parse_repairs_are_bounded_and_errors_sent(tmp_path, cfg):
    response = json.loads((PROJECT_ROOT / "tests/fixtures/mock_responses.json").read_text())["summarizer"]
    transport = ScriptedTransport(["not json", "{}", json.dumps(response)])
    client = make_client(tmp_path, cfg, transport)
    result = client.request("summarizer", {"x": "원문"}, Brief,
                            context={"turn": 0, "condition": "incremental", "essay_id": "mock"}, k=4)
    assert result.issues == []
    assert transport.call_count == 3
    assert [len(messages) for messages in transport.seen] == [2, 4, 6]
    assert "Invalid JSON" in transport.seen[1][-1]["content"]
    assert "Field required" in transport.seen[2][-1]["content"]
    calls = [json.loads(line) for line in client.call_log.read_text().splitlines()]
    assert [row["status"] for row in calls] == ["invalid", "invalid", "ok"]
    assert [row["raw_response"] for row in calls[:2]] == ["not json", "{}"]
    assert len(list(client.cache.glob("*.json"))) == 1


def test_invalid_outputs_fail_after_two_retries(tmp_path, cfg):
    transport = ScriptedTransport(["{}"] * 4)
    client = make_client(tmp_path, cfg, transport)
    with pytest.raises(ValueError, match="after 3 attempts"):
        client.request("summarizer", {}, Brief, context={"turn": 0}, k=4)
    assert transport.call_count == 3
    assert len(client.call_log.read_text().splitlines()) == 3
    assert not list(client.cache.glob("*.json"))


def test_issue_limit_validation_uses_same_bounded_parse_retry(tmp_path, cfg):
    issue = {"id": "i1", "priority": 1, "rubric": "expression_2", "diagnostic_item": None,
             "location": ["s1"], "evidence": "원문", "action": "테스트", "recurring": False}
    good = {"issues": [issue], "keep_style": "유지", "connect_note": None, "learner_feedback": "테스트"}
    bad = {**good, "issues": [issue, {**issue, "id": "i2", "priority": 2}]}
    transport = ScriptedTransport([json.dumps(bad), json.dumps(good)])
    evidence = Evidence(turn=0, rubric_scores={"expression_2": 5.0}, rubric_deltas={"expression_2": None},
                        rubric_rationales=None, weak_rubrics=["expression_2"], diagnostic_items={})
    result = summarize(make_client(tmp_path, cfg, transport), "원문", evidence, [], [], Profile(prompt="문항"),
                       None, k=1, context={"condition": "incremental", "turn": 0})
    assert len(result.issues) == 1 and transport.call_count == 2
    assert "최대 1개" in transport.seen[0][0]["content"]


def test_cache_changes_with_model_messages_and_params(tmp_path, cfg):
    transport = MockTransport(PROJECT_ROOT / "tests/fixtures/mock_responses.json")
    client = make_client(tmp_path, cfg, transport)
    request = lambda payload: client.request("summarizer", payload, Brief, context={"turn": 0}, k=4)
    request({"x": "A"})
    request({"x": "A"})
    assert transport.call_count == 1
    request({"x": "B"})
    cfg.llm.agent.model = "mock-agent-second"
    request({"x": "A"})
    cfg.llm.agent.temperature = 0.1
    request({"x": "A"})
    assert transport.call_count == 4

