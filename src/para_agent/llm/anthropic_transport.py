"""Claude Messages adapter sharing local validation, repair, and cache behavior."""
from pathlib import Path

from dotenv import dotenv_values

from .errors import ProviderResponseError


class AnthropicTransport:
    provider = "anthropic"

    def __init__(self, model, env_file: Path, *, sdk_client=None, max_calls=228):
        if model.provider != self.provider:
            raise ValueError("Anthropic transport requires provider=anthropic")
        if model.model == "claude-haiku-5-5" and model.temperature is not None:
            raise ValueError("Haiku 5.5 requires temperature=null; omission needs an explicit config choice")
        self.model, self.call_count, self.max_calls = model, 0, max_calls
        self.last_metadata, self.last_usage = {}, {}
        if sdk_client is None:
            values = dotenv_values(env_file)
            key = values.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(f"ANTHROPIC_API_KEY missing from {env_file}")
            from anthropic import Anthropic
            sdk_client = Anthropic(api_key=key, base_url="https://api.anthropic.com",
                                   timeout=model.timeout_s, max_retries=0)
        self.client = sdk_client

    def parameters(self, schema, agent):
        from anthropic import transform_schema
        # The SDK moves unsupported field constraints into descriptions. The common
        # client still validates every original Pydantic constraint after generation.
        params = {"model": self.model.model, "max_tokens": self.model.max_output_tokens,
                  "service_tier": "standard_only",
                  "output_config": {"format": {"type": "json_schema", "schema": transform_schema(schema)}}}
        if self.model.temperature is not None:
            # anthropic >=1 moved deprecated sampling arguments to extra_body.
            params["extra_body"] = {"temperature": self.model.temperature}
        if self.model.reasoning_effort == "none":
            params["thinking"] = {"type": "disabled"}
        elif self.model.reasoning_effort is not None:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"]["effort"] = self.model.reasoning_effort
        return params

    def complete(self, *, agent, messages, params):
        self.last_metadata, self.last_usage = {}, {}
        if not messages or messages[0]["role"] != "system" or any(m["role"] == "system" for m in messages[1:]):
            raise ValueError("Expected a leading system prompt followed by user/assistant messages")
        if self.call_count >= self.max_calls:
            raise RuntimeError("Pilot generation call limit reached")
        self.call_count += 1
        response = self.client.messages.create(system=messages[0]["content"], messages=messages[1:], **params)
        native_usage = response.usage.model_dump()
        cached = native_usage.get("cache_read_input_tokens") or 0
        writes = native_usage.get("cache_creation_input_tokens") or 0
        input_tokens = native_usage["input_tokens"] + cached + writes
        output_tokens = native_usage["output_tokens"]
        # Anthropic input_tokens excludes cached/write tokens; normalize totals for
        # cross-provider accounting and keep the untouched provider usage as well.
        self.last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                           "total_tokens": input_tokens + output_tokens,
                           "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": writes}}
        raw = "".join(block.text for block in response.content if block.type == "text")
        self.last_metadata = {"response_id": response.id, "response_model": response.model,
                              "response_status": response.stop_reason, "raw_response": raw,
                              "raw_output": [block.model_dump() for block in response.content],
                              "usage": native_usage, "service_tier": native_usage.get("service_tier")}
        if response.stop_reason != "end_turn":
            raise ProviderResponseError(f"Claude stop reason: {response.stop_reason}")
        return raw, self.last_usage

    def close(self):
        self.client.close()
