import pytest

from para_agent.diagnose.evidence import build_evidence
from para_agent.diagnose.schema import Diagnosis
from para_agent.memory.schema import SessionMemory


@pytest.mark.parametrize("mode,expected", [("prev", 3.0), ("cum", 4.0)])
def test_delta_excludes_current_and_length_sensitive_features(mode, expected, kb):
    memory = SessionMemory()
    for turn, value in enumerate([2.0, 4.0, 7.0]):
        diag = Diagnosis(unit="para", rubric_scores={"expression_1": value},
                         features={key: value for key in kb.key_map.values()})
        memory.add_turn_diag(turn, diag)
    evidence = build_evidence(diag, memory, delta_mode=mode, weak_rubrics=1,
                             rubric_diag_map={"expression_1": ["표현의 적절성", "표현의 다양성"]}, kb=kb)
    assert evidence.turn == 2
    assert evidence.rubric_deltas["expression_1"] == expected
    observations = {row.code: row for rows in evidence.diagnostic_items.values() for row in rows}
    assert observations["morph_LenAvg"].delta == expected
    assert observations["morph_NDW"].value == 7.0
    assert observations["morph_NDW"].delta is None


def test_missing_previous_value_is_not_zero(kb):
    memory = SessionMemory()
    for turn, value in enumerate([2.0, None, 7.0]):
        diag = Diagnosis(unit="cum", rubric_scores={"expression_1": 5},
                         features={key: value for key in kb.key_map.values()})
        memory.add_turn_diag(turn, diag)
    kwargs = dict(weak_rubrics=1, rubric_diag_map={"expression_1": ["표현의 적절성"]}, kb=kb)
    prev = build_evidence(diag, memory, delta_mode="prev", **kwargs)
    cum = build_evidence(diag, memory, delta_mode="cum", **kwargs)
    assert all(row.delta is None for row in prev.diagnostic_items["표현의 적절성"])
    assert all(row.delta == 5 for row in cum.diagnostic_items["표현의 적절성"])

