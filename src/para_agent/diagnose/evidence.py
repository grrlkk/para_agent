from statistics import mean

from para_agent.kb.lookup import KnowledgeBase
from para_agent.memory.schema import SessionMemory

from .schema import Diagnosis, DiagnosisPair, Evidence, EvidencePair, FeatureObs


def _delta(value, history, field, key, mode):
    if value is None or not history:
        return None
    prior = history[-1:] if mode == "prev" else history
    values = [row[field].get(key) for row in prior]
    values = [item for item in values if item is not None]
    return value - mean(values) if values else None


def build_evidence(diag: Diagnosis | DiagnosisPair, memory: SessionMemory, *, delta_mode: str,
                   weak_rubrics: int, rubric_diag_map: dict, kb: KnowledgeBase) -> Evidence | EvidencePair:
    if delta_mode not in {"prev", "cum"}:
        raise ValueError("delta_mode must be prev or cum")
    if not memory.turn_diags:
        raise ValueError("Store the current diagnosis before building evidence")
    turn = memory.turn_diags[-1]["turn"]
    if memory.turn_diags[-1] != {"turn": turn, **diag.model_dump()}:
        raise ValueError("Stored current diagnosis does not match the evidence input")
    history = memory.turn_diags[:-1]
    kwargs = dict(turn=turn, history=history, delta_mode=delta_mode,
                  weak_rubrics=weak_rubrics, rubric_diag_map=rubric_diag_map, kb=kb)
    if isinstance(diag, DiagnosisPair):
        return EvidencePair(para=_unit_evidence(diag.para, **kwargs),
                            cum=_unit_evidence(diag.cum, **kwargs))
    return _unit_evidence(diag, **kwargs)


def _unit_evidence(diag: Diagnosis, *, turn, history, delta_mode, weak_rubrics,
                   rubric_diag_map, kb) -> Evidence:
    # Keep each previous turn in place, even if that unit is absent: prev must
    # never skip the previous turn or subtract a score from a different unit.
    history = [row.get(diag.unit, row if row.get("unit") == diag.unit else
                       {"rubric_scores": {}, "features": {}}) for row in history]
    weak = sorted(diag.rubric_scores, key=diag.rubric_scores.get)[:weak_rubrics]
    items = dict.fromkeys(item for rubric in weak for item in rubric_diag_map[rubric])
    observations = {item: [] for item in items}
    for entry in kb.lookup(items):
        key = kb.key_map[entry["code"]]
        value = diag.features[key]
        delta = None if entry["length_sensitive"] else _delta(value, history, "features", key, delta_mode)
        observations[entry["diagnostic_item"]].append(FeatureObs(code=key, value=value, delta=delta))
    return Evidence(turn=turn, rubric_scores=diag.rubric_scores,
                    rubric_deltas={key: _delta(value, history, "rubric_scores", key, delta_mode)
                                   for key, value in diag.rubric_scores.items()},
                    rubric_rationales=diag.rubric_rationales, weak_rubrics=weak,
                    diagnostic_items=observations)

