import json
from types import SimpleNamespace

import pytest

from para_agent.agents.reviser import ReviserOutput
from para_agent.agents.summarizer import Brief
from para_agent.config import PROJECT_ROOT, ModelConfig
from para_agent.llm.anthropic_transport import AnthropicTransport
from para_agent.llm.client import Client
from para_agent.llm.errors import ProviderResponseError


def model():
    return ModelConfig(provider="anthropic", model="claude-haiku-5-5", temperature=None, reasoning_effort="none")


def setup(tmp_path, *, stop_reason="end_turn", interruption=False):
    calls = []
    raw = json.dumps({"issues": [], "keep_style": "해라체", "connect_note": None, "learner_feedback": "유지"})
    usage = {"input_tokens": 20, "cache_read_input_tokens": 7, "cache_creation_input_tokens": 3,
             "output_tokens": 10, "service_tier": "standard"}

    def create(**kwargs):
        calls.append(kwargs)
        if interruption:
            raise KeyboardInterrupt()
        return SimpleNamespace(id="msg-test", model="claude-haiku-5-5", stop_reason=stop_reason,
                               usage=SimpleNamespace(model_dump=lambda: usage),
                               content=[SimpleNamespace(type="text", text=raw,
                                        model_dump=lambda: {"type": "text", "text": raw})])

    sdk = SimpleNamespace(messages=SimpleNamespace(create=create), close=lambda: None)
    config = model()
    transport = AnthropicTransport(config, tmp_path / ".env", sdk_client=sdk)
    client = Client(model=config, transport=transport, prompts=PROJECT_ROOT / "prompts",
                    cache=tmp_path / "cache", call_log=tmp_path / "calls.jsonl")
    return transport, client, calls


def test_claude_payload_and_normalized_usage_with_cache_replay(tmp_path):
    transport, client, calls = setup(tmp_path)
    for _ in range(2):
        client.request("summarizer", {"x":"원문"}, Brief, context={"turn":0}, k=4)
    assert len(calls) == transport.call_count == 1
    request = calls[0]
    assert request["system"].startswith("너는 한국어 작문")
    assert [m["role"] for m in request["messages"]] == ["user"]
    assert request["model"] == "claude-haiku-5-5"
    assert request["thinking"] == {"type":"disabled"}
    assert request["service_tier"] == "standard_only"
    assert "temperature" not in request and "extra_body" not in request
    assert request["output_config"]["format"]["type"] == "json_schema"
    rows = [json.loads(line) for line in client.call_log.read_text().splitlines()]
    assert rows[0]["tokens"]["input_tokens"] == 30
    assert rows[0]["tokens"]["total_tokens"] == 40
    assert rows[0]["provider_response"]["usage"]["input_tokens"] == 20
    assert rows[1]["tokens"]["total_tokens"] == 0 and rows[1]["provider_response"]["response_id"] == "msg-test"
    assert [r["real_api_call"] for r in rows] == [True,False]


def test_anthropic_schema_transform_keeps_pydantic_constraints(tmp_path):
    transport, client, calls = setup(tmp_path)
    from para_agent.agents.summarizer import Issue
    schema = Issue.model_json_schema()
    sent = transport.parameters(schema, "issue")["output_config"]["format"]["schema"]
    assert "minimum" not in sent["properties"]["priority"]
    assert "minimum" in sent["properties"]["priority"]["description"]
    assert schema["properties"]["priority"]["minimum"] == 1
    with pytest.raises(ValueError):
        Issue(id="i", priority=0, rubric="task_1", diagnostic_item=None,
              location=["s1"], evidence="", action="", recurring=False)


def test_empty_revision_reaches_local_fallback_without_schema_retry():
    out = ReviserOutput.model_validate({"revised_paragraph":"","applied_issue_ids":[],"skipped":[]})
    assert out.revised_paragraph == ""


@pytest.mark.parametrize("stop_reason", ["max_tokens","refusal","model_context_window_exceeded"])
def test_claude_incomplete_and_refusal_fail_without_repair(tmp_path, stop_reason):
    transport, client, calls = setup(tmp_path, stop_reason=stop_reason)
    with pytest.raises(ProviderResponseError):
        client.request("summarizer", {}, Brief, context={"turn":0}, k=4)
    assert len(calls) == 1
    record = json.loads(client.call_log.read_text())
    assert record["status"] == "error" and record["raw_response"]
    assert record["tokens"]["total_tokens"] == 40
    assert record["provider_response"]["response_status"] == stop_reason


def test_interrupt_records_dispatch_and_propagates_without_retry(tmp_path):
    transport, client, calls = setup(tmp_path, interruption=True)
    with pytest.raises(KeyboardInterrupt):
        client.request("summarizer", {}, Brief, context={"turn":0}, k=4)
    row = json.loads(client.call_log.read_text())
    assert row["status"] == "interrupted" and row["real_api_call"]
    assert len(calls) == 1 and not list(client.cache.glob("*.json"))


def test_claude_no_silent_temperature_change_or_ambient_key(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="temperature=null"):
        AnthropicTransport(model().model_copy(update={"temperature":0}), tmp_path / ".env")
    monkeypatch.setenv("ANTHROPIC_API_KEY","ambient-key-not-used")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY missing"):
        AnthropicTransport(model(), tmp_path / ".env")
