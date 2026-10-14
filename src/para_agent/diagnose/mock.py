from .feature_adapter import MockFeatureExtractor
from .schema import Diagnosis, DiagnosisPair
from .scorer_adapter import MockScorer


class MockDiagnoser:
    def __init__(self, rubric_keys: list[str], feature_keys: list[str]):
        self.scorer = MockScorer(rubric_keys)
        self.extractor = MockFeatureExtractor(feature_keys)

    def diagnose(self, x: str, prefix: list[str], unit: str, prompt: str) -> Diagnosis | DiagnosisPair:
        if unit == "both":
            return DiagnosisPair(para=self.diagnose(x, prefix, "para", prompt),
                                 cum=self.diagnose(x, prefix, "cum", prompt))
        if unit == "hybrid":
            raise NotImplementedError("Selective hybrid is deferred; use both to retain both diagnoses")
        if unit not in {"para", "cum"}:
            raise ValueError(f"Unknown diagnosis unit: {unit}")
        text = x if unit == "para" else "\n\n".join([*prefix, x])
        scores, rationales = self.scorer.score(text, prompt)
        return Diagnosis(unit=unit, rubric_scores=scores, rubric_rationales=rationales,
                         features=self.extractor.extract(text))

