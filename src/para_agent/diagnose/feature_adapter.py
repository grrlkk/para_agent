"""Scope and missing-value adapter; original feature calculations are unchanged."""
import hashlib
import math
from pathlib import Path
import time
from typing import Protocol

from para_agent.artifacts import append_jsonl

from .feature_worker import FeatureWorker


class FeatureExtractor(Protocol):
    def extract(self, text: str) -> dict[str, float | None]: ...


class MockFeatureExtractor:
    def __init__(self, feature_keys: list[str]):
        self.feature_keys = feature_keys

    def extract(self, text: str) -> dict[str, float | None]:
        # No linguistic values are fabricated. Actual computation belongs to stage 1.
        return dict.fromkeys(self.feature_keys, None)


class FeatureAdapter:
    def __init__(self, config, feature_keys, kb, *, log_path: Path, worker=None):
        self.feature_keys, self.log_path = list(feature_keys), log_path
        self.worker = worker if worker is not None else FeatureWorker(config)
        self.scopes = {kb.key_map[row["code"]]: row["unit_scope"] for row in kb.entries}
        if set(self.scopes) != set(self.feature_keys):
            raise ValueError("KB scope mapping must cover the actual 29 feature keys")

    def extract(self, text: str, *, unit: str, paragraph_count: int):
        if unit not in {"para", "cum"} or paragraph_count < 1 or not text.strip():
            raise ValueError("Feature extraction requires text, a valid unit, and a positive paragraph count")
        started = time.perf_counter()
        record = {"component": "features", "unit": unit, "paragraph_count": paragraph_count,
                  "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        try:
            response = self.worker(text)
            details = response["details"]
            if set(details) != set(self.feature_keys):
                raise ValueError("Existing extractor returned different feature keys; no guessing or renaming applied")
            features, missing = {}, {}
            for key in self.feature_keys:
                item = details[key]
                if "imputed" not in item or "value" not in item:
                    raise ValueError(f"Missing provenance for feature {key}")
                scope = self.scopes[key]
                reason = None
                if scope == "cum" and (unit != "cum" or paragraph_count < 2):
                    reason = "cumulative_requires_multiple_paragraphs"
                elif scope == "para" and unit != "para":
                    reason = "paragraph_only"
                elif item["imputed"]:
                    reason = "legacy_imputed_value"
                elif item["value"] is None or not math.isfinite(float(item["value"])):
                    reason = "unavailable_or_nonfinite"
                features[key] = None if reason else float(item["value"])
                if reason:
                    missing[key] = reason
            record.update(status="ok", device=response.get("device"), features=features,
                          unavailable=missing, sources={key: item.get("source") for key, item in details.items()})
            return features
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            record["latency_s"] = time.perf_counter() - started
            append_jsonl(self.log_path, record)

    def close(self):
        close = getattr(self.worker, "close", None)
        if close:
            close()

