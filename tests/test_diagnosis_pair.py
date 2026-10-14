import pytest

from para_agent.diagnose.evidence import build_evidence
from para_agent.diagnose.mock import MockDiagnoser
from para_agent.diagnose.schema import Diagnosis, DiagnosisPair
from para_agent.memory.schema import SessionMemory


def test_both_scores_all_rubrics_on_both_inputs_and_preserves_results(contracts):
    diagnoser = MockDiagnoser(contracts["rubric_keys"], contracts["feature_keys"])
    scored, extracted = [], []

    def score(text, prompt):
        scored.append((text, prompt))
        value = 3.0 if text == "이번 원문" else 7.0
        return dict.fromkeys(contracts["rubric_keys"], value), None

    def extract(text):
        extracted.append(text)
        return dict.fromkeys(contracts["feature_keys"], None)

    diagnoser.scorer.score = score
    diagnoser.extractor.extract = extract
    pair = diagnoser.diagnose("이번 원문", ["앞 수정본"], "both", "문항")
    assert scored == [("이번 원문", "문항"), ("앞 수정본\n\n이번 원문", "문항")]
    assert extracted == ["이번 원문", "앞 수정본\n\n이번 원문"]
    assert set(pair.para.rubric_scores.values()) == {3.0}
    assert set(pair.cum.rubric_scores.values()) == {7.0}
    assert set(pair.model_dump()) == {"para", "cum"}
    first_turn = diagnoser.diagnose("이번 원문", [], "both", "문항")
    assert first_turn.para.rubric_scores == first_turn.cum.rubric_scores
    assert first_turn.para.unit == "para" and first_turn.cum.unit == "cum"


@pytest.mark.parametrize("mode,para_delta,cum_delta", [("prev", 3.0, 1.0), ("cum", 3.5, 2.0)])
def test_two_unit_histories_keep_separate_scores_features_and_deltas(kb, mode, para_delta, cum_delta):
    memory = SessionMemory()
    for turn, (para_score, cum_score) in enumerate([(1.0, 4.0), (2.0, 6.0), (5.0, 7.0)]):
        pair = DiagnosisPair(
            para=Diagnosis(unit="para", rubric_scores={"expression_1": para_score},
                           features={key: para_score for key in kb.key_map.values()}),
            cum=Diagnosis(unit="cum", rubric_scores={"expression_1": cum_score},
                          features={key: cum_score for key in kb.key_map.values()}))
        memory.add_turn_diag(turn, pair)
    evidence = build_evidence(pair, memory, delta_mode=mode, weak_rubrics=1,
                              rubric_diag_map={"expression_1": ["표현의 적절성", "표현의 다양성"]}, kb=kb)
    for view, score, delta in [(evidence.para, 5.0, para_delta), (evidence.cum, 7.0, cum_delta)]:
        assert view.turn == 2
        assert view.rubric_scores == {"expression_1": score}
        assert view.rubric_deltas == {"expression_1": delta}
        features = {row.code: row for values in view.diagnostic_items.values() for row in values}
        assert features["morph_LenAvg"].value == score
        assert features["morph_LenAvg"].delta == delta
        assert features["morph_NDW"].delta is None


def test_delta_does_not_compare_against_another_unit(kb):
    memory = SessionMemory()
    memory.add_turn_diag(0, Diagnosis(unit="para", rubric_scores={"expression_1": 2.0},
                                     features={key: 2.0 for key in kb.key_map.values()}))
    current = Diagnosis(unit="cum", rubric_scores={"expression_1": 7.0},
                        features={key: 7.0 for key in kb.key_map.values()})
    memory.add_turn_diag(1, current)
    evidence = build_evidence(current, memory, delta_mode="prev", weak_rubrics=1,
                              rubric_diag_map={"expression_1": ["표현의 적절성"]}, kb=kb)
    assert evidence.rubric_deltas == {"expression_1": None}
    assert all(row.delta is None for row in evidence.diagnostic_items["표현의 적절성"])
