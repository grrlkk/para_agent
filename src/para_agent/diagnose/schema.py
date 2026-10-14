from typing import Literal

from pydantic import model_validator

from para_agent.schema import Schema


class Diagnosis(Schema):
    unit: Literal["para", "cum"]
    rubric_scores: dict[str, float]
    rubric_rationales: dict[str, str] | None = None
    features: dict[str, float | None]


class DiagnosisPair(Schema):
    """Keep the two native diagnoses intact; do not select or merge rubric scores."""
    para: Diagnosis
    cum: Diagnosis

    @model_validator(mode="after")
    def matching_units(self):
        if self.para.unit != "para" or self.cum.unit != "cum":
            raise ValueError("DiagnosisPair must contain the named para and cum units")
        if self.para.rubric_scores.keys() != self.cum.rubric_scores.keys():
            raise ValueError("Both diagnoses must use the same rubric keys")
        if self.para.features.keys() != self.cum.features.keys():
            raise ValueError("Both diagnoses must use the same feature keys")
        return self


class FeatureObs(Schema):
    code: str
    value: float | None
    delta: float | None


class Evidence(Schema):
    turn: int
    rubric_scores: dict[str, float]
    rubric_deltas: dict[str, float | None]
    rubric_rationales: dict[str, str] | None
    weak_rubrics: list[str]
    diagnostic_items: dict[str, list[FeatureObs]]


class EvidencePair(Schema):
    para: Evidence
    cum: Evidence


def evidence_views(evidence: Evidence | EvidencePair) -> list[Evidence]:
    return [evidence.para, evidence.cum] if isinstance(evidence, EvidencePair) else [evidence]

