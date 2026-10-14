"""Thin adapter around FEAK-Agent's unchanged deterministic KananaScorer."""
import hashlib
import importlib
import math
from pathlib import Path
import sys
import time
from typing import Protocol

import yaml

from para_agent.artifacts import append_jsonl, file_hash
from para_agent.config import project_path


class Scorer(Protocol):
    def score(self, text: str, prompt: str) -> tuple[dict[str, float], dict[str, str] | None]: ...


class MockScorer:
    def __init__(self, rubric_keys: list[str]):
        self.rubric_keys = rubric_keys

    def score(self, text: str, prompt: str) -> tuple[dict[str, float], None]:
        # Deliberately fixed fixture values, never measurements or estimated scores.
        return dict.fromkeys(self.rubric_keys, 5.0), None


def load_scorer_settings(config):
    source = Path(config.source_config)
    settings = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = Path(settings["paths"]["repo"])
    settings["paths"] = {key: Path(value) if Path(value).is_absolute() else root / value
                         for key, value in settings["paths"].items()}
    # Keep all inference/model settings; use project-owned caches and approved metadata.
    settings["paths"]["output"] = project_path(config.cache_dir)
    if config.metadata_dir is not None:
        from .scorer_metadata import validate_metadata_override
        directory = project_path(config.metadata_dir)
        validate_metadata_override(settings["paths"]["metadata"] / "genres.json", directory / "genres.json")
        settings["paths"]["metadata"] = directory
    return settings


class KananaScorerAdapter:
    def __init__(self, config, rubric_keys, *, log_path: Path, backend=None):
        self.config, self.rubric_keys, self.log_path = config, list(rubric_keys), log_path
        self.backend = backend
        self.settings = load_scorer_settings(config)
        metadata = self.settings["paths"]["metadata"] / "genres.json"
        self.metadata_provenance = {"path": str(metadata), "sha256": file_hash(metadata)}
        if config.backend != "verak.v3.score.kanana.KananaScorer":
            raise ValueError("Use the user-confirmed FEAK-Agent KananaScorer backend")

    def _ensure_backend(self):
        if self.backend is None:
            root = str(Path(self.config.repository_path).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            module = importlib.import_module("verak.v3.score.kanana")
            if not Path(module.__file__).resolve().is_relative_to(Path(root)):
                raise RuntimeError("Imported scorer is outside the configured FEAK-Agent repository")
            self.backend = module.KananaScorer(self.settings)
        return self.backend

    def score(self, text: str, prompt: str, *, use_cache: bool = True):
        started = time.perf_counter()
        record = {"component": "scorer", "backend": self.config.backend,
                  "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "use_cache": use_cache, "truncated": False,
                  "genre_metadata": self.metadata_provenance,
                  "max_input_tokens": self.settings["scorer"]["max_input_tokens"]}
        try:
            native = self._ensure_backend()
            result = native.score(prompt, text, use_cache=use_cache).to_dict()
            scores = result["expected"]
            if len(scores) != len(self.rubric_keys) or not all(math.isfinite(value) for value in scores):
                raise ValueError("Kanana returned invalid expected scores; no replacement or clipping applied")
            record.update(status="ok", result=result, model_fingerprint=native.fingerprint)
            # This existing path intentionally returns score-only output, with no rationales.
            return dict(zip(self.rubric_keys, scores)), None
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__, error=str(exc))
            if hasattr(exc, "tokens") and hasattr(exc, "limit"):
                record.update(status="input_too_long", input_tokens=exc.tokens,
                              max_input_tokens=exc.limit, overflow_policy="reject_without_truncation")
            raise
        finally:
            record["latency_s"] = time.perf_counter() - started
            append_jsonl(self.log_path, record)

    def close(self):
        if self.backend is not None:
            self.backend.close()

