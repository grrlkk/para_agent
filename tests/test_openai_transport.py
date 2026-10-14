import json
from types import SimpleNamespace

import pytest

from para_agent.agents.summarizer import Brief
from para_agent.config import PROJECT_ROOT, ModelConfig
from para_agent.llm.client import Client
from para_agent.llm.openai_transport import OpenAITransport, ProviderResponseError, strict_schema
from para_agent.memory.schema import MemoryUpdate


def provider_response(raw, *, status="completed", refusal=False):
    content = ([{"type": "refusal", "refusal": "refused"}] if refusal else
               [{"type": "output_text", "text": raw}])
    usage = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
             "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 0},
             "output_tokens_details": {"reasoning_tokens": 0}}
    return SimpleNamespace(id="response-test", model="gpt-6-luna", status=status,
                           service_tier="default", output_text=raw,
                           usage=SimpleNamespace(model_dump=lambda: usage),
                           output=[SimpleNamespace(model_dump=lambda: {"type": "message", "content": content})],
                           incomplete_details=(SimpleNamespace(model_dump=lambda: {"reason": "max_output_tokens"})
                                               if status == "incomplete" else None))


def model():
    return ModelConfig(provider="openai", model="gpt-6-luna", temperature=0, reasoning_effort="none")


def setup(tmp_path, response):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if isinstance(response, Exception):
            raise response
        return response

    sdk = SimpleNamespace(responses=SimpleNamespace(create=create), close=lambda: None)
    transport = OpenAITransport(model(), tmp_path / ".env", sdk_client=sdk, max_calls=3)
    client = Client(model=model(), transport=transport, prompts=PROJECT_ROOT / "prompts",
                    cache=tmp_path / "cache", call_log=tmp_path / "calls.jsonl")
    return transport, client, calls


def test_responses_payload_and_cache_preserve_metadata_without_charging_again(tmp_path):
    raw = json.dumps({"issues": [], "keep_style": "해라체", "connect_note": None, "learner_feedback": "유지"})
    transport, client, calls = setup(tmp_path, provider_response(raw))
    for _ in range(2):
        client.request("summarizer", {"x": "원문"}, Brief, context={"turn": 0}, k=4)
    assert transport.call_count == len(calls) == 1
    sent = calls[0]
    assert sent["model"] == "gpt-6-luna" and sent["temperature"] == 0
    assert sent["reasoning"] == {"effort": "none"}
    assert sent["store"] is False and sent["truncation"] == "disabled"
    assert sent["service_tier"] == "default"
    assert sent["text"]["format"]["strict"] is True
    assert len(sent["input"]) == 2
    rows = [json.loads(line) for line in client.call_log.read_text().splitlines()]
    assert [r["real_api_call"] for r in rows] == [True, False]
    assert [r["tokens"]["total_tokens"] for r in rows] == [30, 0]
    assert all(r["mock"] is False and r["raw_response"] == raw for r in rows)
    assert rows[1]["provider_response"]["response_id"] == "response-test"
    assert rows[1]["provider_response"]["usage"]["total_tokens"] == 30
    client.model.max_output_tokens += 1
    # The transport owns the same validated config in production.
    transport.model.max_output_tokens = client.model.max_output_tokens
    client.request("summarizer", {"x": "원문"}, Brief, context={"turn": 0}, k=4)
    assert transport.call_count == 2


def test_strict_nullable_profile_schema_does_not_mutate_local_validation():
    schema = MemoryUpdate.model_json_schema()
    strict = strict_schema(schema)
    patch = strict["$defs"]["ProfilePatch"]
    assert set(patch["required"]) == {"topic", "style_notes", "thesis"}
    assert 'register' not in patch['properties']
    assert patch["additionalProperties"] is False
    assert "default" not in patch["properties"]["topic"]
    assert {"type": "null"} in patch["properties"]["topic"]["anyOf"]
    assert "default" in schema["$defs"]["ProfilePatch"]["properties"]["topic"]
    partial = MemoryUpdate.model_validate({"error_notes": [], "profile_update": {}, "turn_summary": "유지"})
    assert partial.profile_update.model_fields_set == set()


@pytest.mark.parametrize("status,refusal", [("incomplete", False), ("completed", True)])
def test_incomplete_and_refusal_are_logged_once_without_schema_repair(tmp_path, status, refusal):
    transport, client, calls = setup(tmp_path, provider_response("partial", status=status, refusal=refusal))
    with pytest.raises(ProviderResponseError):
        client.request("summarizer", {}, Brief, context={"turn": 0}, k=4)
    assert len(calls) == 1
    record = json.loads(client.call_log.read_text())
    assert record["status"] == "error" and record["tokens"]["total_tokens"] == 30
    assert record["provider_response"]["raw_output"] and record["raw_response"] == "partial"
    assert not list(client.cache.glob("*.json"))


def test_provider_exception_does_not_retry_or_log_sensitive_message(tmp_path):
    transport, client, calls = setup(tmp_path, RuntimeError("do-not-log-test-secret"))
    with pytest.raises(RuntimeError):
        client.request("summarizer", {}, Brief, context={"turn": 0}, k=4)
    log = client.call_log.read_text()
    assert "do-not-log-test-secret" not in log
    assert json.loads(log)["real_api_call"] is True
    assert len(calls) == 1


def test_live_credentials_must_come_from_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-key-is-not-used")
    with pytest.raises(ValueError, match="OPENAI_API_KEY missing"):
        OpenAITransport(model(), tmp_path / ".env")


def test_call_cap_does_not_record_an_unsent_call(tmp_path):
    transport, client, calls = setup(tmp_path, provider_response("{}"))
    transport.max_calls = 0
    with pytest.raises(RuntimeError, match="call limit"):
        client.request("summarizer", {}, Brief, context={"turn": 0}, k=4)
    assert calls == [] and json.loads(client.call_log.read_text())["real_api_call"] is False


def test_gpt6_temperature_is_not_silently_dropped(tmp_path):
    bad = model().model_copy(update={"reasoning_effort": "low"})
    with pytest.raises(ValueError, match="temperature requires"):
        OpenAITransport(bad, tmp_path / ".env")
