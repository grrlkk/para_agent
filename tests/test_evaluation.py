from types import SimpleNamespace

import pytest

from para_agent.eval.aggregate import aggregate, call_costs, paired_statistics
from para_agent.eval.edit_metrics import edit_metrics, levenshtein
from para_agent.eval.final_score import score_essays


def test_edit_metrics_preserve_korean_whitespace_and_normalize_by_longer_text():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("가 나\n", "가나") == 2
    result = edit_metrics("가나", "가나다라")
    assert result["normalized_edit_distance"] == 0.5
    assert result["length_ratio"] == 2
    assert edit_metrics("가나", "")["normalized_edit_distance"] == 1
    with pytest.raises(ValueError):
        edit_metrics("", "내용")


def test_statistics_exact_sign_test_ties_zeros_and_paired_bootstrap():
    result = paired_statistics([1, 1, 1, 1, 1])
    assert result["wilcoxon_p_two_sided"] == 0.0625
    assert result["ci95"] == [1, 1]
    assert paired_statistics([0, 1, -1])["wilcoxon_p_two_sided"] == 1
    assert paired_statistics([0, 0])["wilcoxon_p_two_sided"] == 1
    assert paired_statistics([1, 2, 3]) == paired_statistics([1, 2, 3])


def test_scoring_uses_same_scorer_for_original_never_human_labels(tmp_path):
    essay = SimpleNamespace(essay_id="example", prompt="질문", paragraphs=["가", "나"],
                            original_scores={"verified_key": 999})
    finals = {(c, "example"): {"final_text": "가\n\n나!", "Y": ["가", "나!"]}
              for c in ("whole_once", "incremental")}
    calls = []
    def score(text, prompt):
        calls.append((text, prompt))
        return {"verified_key": float(len(text))}, None
    rows = score_essays([essay], finals, SimpleNamespace(score=score), ["verified_key"], tmp_path)
    assert calls == [("가\n\n나", "질문"), ("가\n\n나!", "질문"), ("가\n\n나!", "질문")]
    assert [r["delta_total"] for r in rows] == [0, 1, 1]
    with pytest.raises(FileExistsError):
        score_essays([essay], finals, SimpleNamespace(score=score), ["verified_key"], tmp_path)


def test_aggregation_matches_ids_not_row_positions_and_rejects_missing_pairs():
    rows = [{"essay_id": e, "condition": c, "k": v, "total": v}
            for e, c, v in [("b", "incremental", 5), ("a", "original", 1),
                            ("b", "original", 2), ("a", "whole_once", 2),
                            ("a", "incremental", 3), ("b", "whole_once", 3)]]
    result = aggregate(rows, ["k"])
    assert result["incremental_minus_whole_once"]["k"]["mean_difference"] == 1.5
    with pytest.raises(ValueError):
        aggregate(rows[:-1], ["k"])
    with pytest.raises(ValueError):
        aggregate(rows + [rows[0]], ["k"])


def test_usage_counts_failed_attempts_and_does_not_count_cached_original_usage():
    rows = [{"essay_id": "a", "condition": "whole_once", "status": status,
             "real_api_call": live, "cache_hit": not live, "attempt": attempt,
             "latency_s": 1, "tokens": {"input_tokens": tokens, "total_tokens": tokens},
             "provider_response": {"usage": {"input_tokens": 9999}}}
            for status, live, attempt, tokens in [("invalid", True, 0, 10),
                                                 ("ok", True, 1, 20), ("ok", False, 0, 0)]]
    result = call_costs(rows, "a", "whole_once")
    assert result["real_api_calls"] == 2 and result["input_tokens"] == 30
    assert result["cache_hits"] == 1 and result["parse_repairs"] == 1
