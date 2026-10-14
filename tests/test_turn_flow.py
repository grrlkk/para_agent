import json

import pytest
import yaml

from para_agent.agents.reviser import ReviserOutput
from para_agent.agents.summarizer import Brief
from para_agent.config import PROJECT_ROOT
from para_agent.data.loader import Essay
from para_agent.diagnose.mock import MockDiagnoser
from para_agent.memory.schema import MemoryUpdate
from para_agent.run.common import Services
from para_agent.run.incremental import run


class RecordingClient:
    is_mock = True

    def __init__(self):
        self.calls = []

    def request(self, agent, payload, schema, *, context, **kwargs):
        # Serialize now so later in-place profile updates cannot change the captured input.
        from para_agent.artifacts import json_text
        self.calls.append((agent, json.loads(json_text(payload))))
        turn = context["turn"]
        if agent == "summarizer":
            return Brief(issues=[], keep_style="해라체", connect_note=None, learner_feedback="테스트")
        if agent == "reviser":
            return ReviserOutput(revised_paragraph=f"수정 {turn}. 앞부분의 문체도 참고한다. 다음 문장이다. 끝 문장이다.",
                                 applied_issue_ids=[], skipped=[])
        return MemoryUpdate.model_validate({"error_notes": [{"turn": turn, "type": "조사", "example": "원문", "fix": "수정"}],
                                            "profile_update": {"style_notes": "해라체 문장"}, "turn_summary": "조사 수정"})


@pytest.mark.parametrize("use_memory", [True, False])
def test_revised_prefix_prev_tail_and_memory_handoff(tmp_path, cfg, contracts, kb, use_memory):
    cfg.use_memory = use_memory
    cfg.reviser_input = "prev_plus_x"
    essay = Essay(essay_id="flow", prompt="문항", paragraphs=["원문 0.", "원문 1.", "원문 2."], original_scores={})
    client = RecordingClient()
    services = Services(MockDiagnoser(contracts["rubric_keys"], contracts["feature_keys"]), client, kb,
                        yaml.safe_load((PROJECT_ROOT / "configs/rubric_diag_map.yaml").read_text()), tmp_path)
    final = run(essay, cfg, services)
    assert [name for name, _ in client.calls] == ["summarizer", "reviser", "memory_update"] * 3
    expected = [f"수정 {turn}. 앞부분의 문체도 참고한다. 다음 문장이다. 끝 문장이다." for turn in range(3)]
    for turn in range(3):
        summary, revision = client.calls[turn * 3][1], client.calls[turn * 3 + 1][1]
        prev = expected[turn - 1] if turn else None
        assert summary["prev"] == prev
        assert revision["rev_in"] == {"prev": prev, "x": essay.paragraphs[turn]}
        # The full revised paragraph must survive, even when the boundary hint is shorter.
        assert revision["prev_tail"] == ("다음 문장이다. 끝 문장이다." if turn else None)
        assert revision["brief"]["keep_style"] == "해라체"
        assert len(summary["memory_notes"]) == (turn if use_memory else 0)
        assert summary["profile"]["register"] == ("해라체" if use_memory and turn else None)
        assert revision["profile"] == summary["profile"]
        record = json.loads((tmp_path / f"incremental/flow/turn_{turn}.json").read_text())
        assert record["prefix"] == expected[:turn]
    assert final["Y"] == expected
    assert len(final["memory"].error_notes) == (3 if use_memory else 0)

