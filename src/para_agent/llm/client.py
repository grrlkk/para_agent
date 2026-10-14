"""Structured-output validation, bounded JSON repairs, caching, and call records."""
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel, ValidationError

from para_agent.artifacts import append_jsonl, file_hash, json_text
from para_agent.config import ModelConfig


class MockTransport:
    provider = "mock"

    def __init__(self, fixture: Path):
        self.responses = json.loads(fixture.read_text(encoding="utf-8"))
        self.fingerprint = file_hash(fixture)
        self.call_count = 0

    def parameters(self, schema, agent):
        return {"response_schema": schema, "mock_fixture_sha256": self.fingerprint}

    def complete(self, *, agent, messages, params):
        self.call_count += 1
        payload = json.loads(messages[1]["content"])
        response = deepcopy(self.responses[agent])
        if agent == "reviser":
            rev_in = payload["rev_in"]
            response["revised_paragraph"] = rev_in if isinstance(rev_in, str) else rev_in["x"]
        return json_text(response), {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


class Client:
    def __init__(self, *, model: ModelConfig, transport, prompts: Path,
                 cache: Path, call_log: Path):
        if model.provider != transport.provider:
            raise ValueError("Model provider and transport must agree")
        self.model, self.transport, self.prompts = model, transport, prompts
        self.cache, self.call_log = cache, call_log
        cache.mkdir(parents=True, exist_ok=True)

    @property
    def is_mock(self):
        return self.model.provider == "mock"

    def request(self, agent: str, payload: dict, schema: type[BaseModel], *, context: dict,
                k: int | None = None, validate=None):
        prompt_file = self.prompts / f"{agent}.md"
        system = prompt_file.read_text(encoding="utf-8")
        if k is not None:
            system = system.replace("{k}", str(k))
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json_text(payload)}]
        params = self.transport.parameters(schema.model_json_schema(), agent)
        if self.is_mock:
            params["temperature"] = self.model.temperature
        for attempt in range(3):  # initial request plus at most two parse/schema repairs (§4.1)
            key = hashlib.sha256(json_text([self.model.provider, self.model.model, messages, params]).encode()).hexdigest()
            path = self.cache / f"{key}.json"
            started = time.perf_counter()
            hit = path.is_file()
            raw, usage = None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            metadata = {}
            dispatched = False
            before_calls = self.transport.call_count
            record = {**context, "agent": agent, "model": self.model.model, "provider": self.model.provider,
                      "mock": self.is_mock, "attempt": attempt, "cache_hit": hit, "cache_key": key,
                      "prompt_sha256": file_hash(prompt_file), "messages": deepcopy(messages), "params": params}
            try:
                if hit:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    raw = cached["raw_response"]
                    metadata = cached.get("provider_response", {})
                else:
                    dispatched = True
                    raw, usage = self.transport.complete(agent=agent, messages=messages, params=params)
                    metadata = getattr(self.transport, "last_metadata", {})
                result = schema.model_validate_json(raw)
                if validate:
                    validate(result)
                if not hit:
                    path.write_text(json_text({"raw_response": raw, "provider_response": metadata}), encoding="utf-8")
                record["status"] = "ok"
                return result
            except (ValidationError, ValueError) as exc:
                if raw is None:  # Setup/provider failures are not schema failures.
                    record.update(status="error", error_type=type(exc).__name__)
                    raise
                record.update(status="invalid", error=str(exc))
                if attempt == 2:
                    raise ValueError(f"{agent}: structured output failed after 3 attempts") from exc
                repair_file = self.prompts / "repair.md"
                record["repair_prompt_sha256"] = file_hash(repair_file)
                messages.extend([{"role": "assistant", "content": raw or ""},
                                 {"role": "user", "content": repair_file.read_text(encoding="utf-8").replace("{error}", str(exc))}])
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__,
                              http_status=getattr(exc, "status_code", None),
                              error_code=getattr(exc, "code", None), error_param=getattr(exc, "param", None))
                raise
            except KeyboardInterrupt:
                record.update(status="interrupted", error_type="KeyboardInterrupt")
                raise
            finally:
                if dispatched and not self.is_mock:
                    metadata = getattr(self.transport, "last_metadata", {})
                    usage = getattr(self.transport, "last_usage", {}) or usage
                    if raw is None:
                        raw = metadata.get("raw_response")
                record.update(raw_response=raw, tokens=usage, provider_response=metadata,
                              real_api_call=not self.is_mock and self.transport.call_count > before_calls,
                              latency_s=time.perf_counter() - started)
                append_jsonl(self.call_log, record)

