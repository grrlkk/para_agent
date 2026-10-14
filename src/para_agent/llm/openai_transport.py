"""Responses API transport. Credentials are read only from the configured .env."""
from copy import deepcopy
from pathlib import Path

from dotenv import dotenv_values

from .errors import ProviderResponseError


def strict_schema(schema):
    """Require nullable fields as well, as required by OpenAI Structured Outputs."""
    result = deepcopy(schema)

    def visit(node):
        if isinstance(node, list):
            for value in node:
                visit(value)
        elif isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            for value in node.values():
                visit(value)

    visit(result)
    return result


class OpenAITransport:
    provider = "openai"

    def __init__(self, model, env_file: Path, *, sdk_client=None, max_calls=228):
        if model.provider != self.provider:
            raise ValueError("OpenAI transport requires provider=openai")
        if model.model.startswith("gpt-6") and model.temperature is not None and model.reasoning_effort != "none":
            raise ValueError("GPT-6 temperature requires reasoning_effort=none; do not silently drop it")
        self.model, self.call_count, self.max_calls = model, 0, max_calls
        self.last_metadata, self.last_usage = {}, {}
        if sdk_client is None:
            values = dotenv_values(env_file)
            key = values.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(f"OPENAI_API_KEY missing from {env_file}")
            from openai import OpenAI
            # No hidden SDK retries: every dispatched generation is counted and logged.
            sdk_client = OpenAI(api_key=key, base_url="https://api.openai.com/v1",
                                timeout=model.timeout_s, max_retries=0)
        self.client = sdk_client

    def parameters(self, schema, agent):
        params = {"model": self.model.model, "max_output_tokens": self.model.max_output_tokens,
                  "truncation": "disabled", "store": False, "service_tier": "default",
                  "text": {"format": {"type": "json_schema", "name": agent,
                                      "strict": True, "schema": strict_schema(schema)}}}
        if self.model.temperature is not None:
            params["temperature"] = self.model.temperature
        if self.model.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.model.reasoning_effort}
        return params

    def complete(self, *, agent, messages, params):
        self.last_metadata, self.last_usage = {}, {}
        if self.call_count >= self.max_calls:
            raise RuntimeError("Pilot generation call limit reached")
        self.call_count += 1
        response = self.client.responses.create(input=messages, **params)
        self.last_usage = response.usage.model_dump() if response.usage else {}
        self.last_metadata = {"response_id": response.id, "response_model": response.model,
                              "raw_response": response.output_text,
                              "response_status": response.status, "usage": self.last_usage,
                              "service_tier": response.service_tier,
                              "raw_output": [item.model_dump() for item in response.output],
                              "incomplete_details": (response.incomplete_details.model_dump()
                                                     if response.incomplete_details else None)}
        if response.status != "completed":
            raise ProviderResponseError(f"Provider response status: {response.status}")
        if any(part.get("type") == "refusal" for item in self.last_metadata["raw_output"]
               for part in item.get("content", [])):
            raise ProviderResponseError("Provider refused the request")
        return response.output_text, self.last_usage

    def close(self):
        self.client.close()
