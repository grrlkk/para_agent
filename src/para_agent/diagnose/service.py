"""Compose live scorer and feature adapters for paragraph, cumulative, and both views."""
import re
from typing import Protocol

from .schema import Diagnosis, DiagnosisPair


class DiagnosisService(Protocol):
    def diagnose(self, x: str, prefix: list[str], unit: str,
                 prompt: str) -> Diagnosis | DiagnosisPair: ...


def paragraph_count(text: str) -> int:
    return len([part for part in re.split(r"\n[ \t]*\n", text) if part.strip()])


class Diagnoser:
    def __init__(self, scorer, features):
        self.scorer, self.features = scorer, features

    def diagnose(self, x: str, prefix: list[str], unit: str, prompt: str, *, use_cache=True):
        if not x.strip() or not prompt.strip() or any(not part.strip() for part in prefix):
            raise ValueError("Diagnosis requires nonblank text, prompt, and prefix paragraphs")
        if unit == "both":
            return DiagnosisPair(para=self.diagnose(x, prefix, "para", prompt, use_cache=use_cache),
                                 cum=self.diagnose(x, prefix, "cum", prompt, use_cache=use_cache))
        if unit not in {"para", "cum"}:
            raise ValueError("Use para, cum, or both; selective hybrid is deferred")
        text = x if unit == "para" else "\n\n".join([*prefix, x])
        scores, rationales = self.scorer.score(text, prompt, use_cache=use_cache)
        features = self.features.extract(text, unit=unit, paragraph_count=paragraph_count(text))
        return Diagnosis(unit=unit, rubric_scores=scores, rubric_rationales=rationales, features=features)

    def close(self):
        try:
            self.features.close()
        finally:
            self.scorer.close()


def build_diagnoser(config, *, log_path):
    """Resolve keys from legacy sources before constructing lazy model adapters."""
    import yaml
    from para_agent.config import project_path
    from para_agent.kb.lookup import KnowledgeBase
    from para_agent.kb.validate import validate_kb
    from .contracts import read_legacy_contract
    from .feature_adapter import FeatureAdapter
    from .scorer_adapter import KananaScorerAdapter

    contract = read_legacy_contract(config.features.package_path)
    frozen = yaml.safe_load(project_path(config.paths.contracts).read_text(encoding="utf-8"))
    for key in ("rubric_keys", "feature_keys"):
        if contract[key] != frozen[key]:
            raise ValueError(f"Existing {key} differ from the recorded contract; confirm before proceeding")
    kb_path, map_path = project_path(config.kb.features), project_path(config.kb.feature_key_map)
    validation = validate_kb(kb_path, map_path, config.features.package_path)
    if not validation["ok"]:
        raise ValueError(f"KB validation failed: {validation['errors']}")
    kb = KnowledgeBase(kb_path, map_path)
    return Diagnoser(KananaScorerAdapter(config.scorer, contract["rubric_keys"], log_path=log_path),
                     FeatureAdapter(config.features, contract["feature_keys"], kb, log_path=log_path))
