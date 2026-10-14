"""Experimental sampling adapter around the unchanged FEAK-Agent Kanana model.

This adapter is deliberately not registered in the production diagnosis factory.
It uses full-vocabulary temperature-1 sampling, accepts the first N valid score
lines, and also evaluates their raw nine-digit teacher-forced expectations.
"""
import hashlib
import importlib
import math
import time

import numpy as np


def essay_seed(essay_id: str, master_seed: int = 15042) -> int:
    """Pair variants of the same essay using the same random stream."""
    value = hashlib.sha256(f"{master_seed}:{essay_id}".encode()).digest()
    return int.from_bytes(value[:4], "big") % (2**31 - 1)


def summarize_samples(samples: list[dict], checkpoints=(8, 16, 32, 64), *, methods=('integer_mean','expected_mean')) -> dict:
    if not samples or max(checkpoints) > len(samples):
        raise ValueError("Every requested prefix needs enough accepted samples")
    output = {}
    for n in checkpoints:
        if n < 2:
            raise ValueError("At least two samples are needed to estimate Monte Carlo error")
        summaries = {}
        for method, field in (("integer_mean", "integers"), ("expected_mean", "expected")):
            if method not in methods:
                continue
            values = np.asarray([s[field] for s in samples[:n]], dtype=np.float64)
            if values.shape != (n, 8) or not np.isfinite(values).all():
                raise ValueError("Expected eight finite rubric values in every sample")
            if np.any(values < 1) or np.any(values > 9):
                raise ValueError("Rubric values must be on the unchanged 1–9 scale")
            totals = values.sum(axis=1)
            means = values.mean(axis=0)
            summaries[method] = {
                "rubrics": means.tolist(), "total": float(means.sum()),
                "rubric_mc_se": (values.std(axis=0, ddof=1) / math.sqrt(n)).tolist(),
                # Preserve covariance between the eight rubrics within each line.
                "total_mc_se": float(totals.std(ddof=1) / math.sqrt(n)),
            }
        output[str(n)] = summaries
    return output


def primary_decision(lf_deltas, contraction_deltas, greedy_spearman, smoothed_spearman):
    """Predeclared rule: N=32 integer means, rubric-sum differences."""
    if not lf_deltas or not contraction_deltas:
        raise ValueError("Both invariance comparison families are required")
    values = [*lf_deltas, *contraction_deltas, greedy_spearman, smoothed_spearman]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Cannot decide from missing or nonfinite measurements")
    lf_max = max(abs(float(v)) for v in lf_deltas)
    contraction_max = max(abs(float(v)) for v in contraction_deltas)
    checks = {"lf_max_abs_total_delta_at_most_1": lf_max <= 1.0,
              "contraction_max_abs_total_delta_at_most_1": contraction_max <= 1.0,
              "total_spearman_not_lower": smoothed_spearman >= greedy_spearman}
    return {"method": "integer_mean", "n": 32, "score": "sum_of_8_rubrics",
            "lf_max_abs_total_delta": lf_max, "contraction_max_abs_total_delta": contraction_max,
            "greedy_total_spearman": greedy_spearman, "smoothed_total_spearman": smoothed_spearman,
            "checks": checks, "passes": all(checks.values()),
            "recommended_role": "primary_pending_user_approval" if all(checks.values()) else "auxiliary",
            "existing_evaluation_settings_changed": False}


class SmoothedKananaAdapter:
    def __init__(self, native, *, batch_size=8, expected_batch_size=None,
                 temperature=1.0, max_attempt_multiplier=4, include_expected=True):
        if batch_size < 1 or temperature <= 0 or max_attempt_multiplier < 1:
            raise ValueError("Invalid sampling parameters")
        self.native = native
        self.batch_size = batch_size
        self.expected_batch_size = expected_batch_size or batch_size
        self.temperature = temperature
        self.max_attempt_multiplier = max_attempt_multiplier
        self.include_expected = include_expected
        self.module = importlib.import_module("verak.v3.score.kanana")

    def expected_for_lines(self, prompt, text, lines, *, batch_size=None):
        """Same teacher forcing and digit softmax; compute only needed LM-head rows.

        logits_to_keep only omits unused output projections; every context token
        still passes through the original transformer. No prefix is shortened.
        """
        import torch
        native = self.native
        encoded = native.prepare_input(prompt, text)
        prefix = encoded["input_ids"][0].tolist()
        length = len(prefix)
        all_results = []
        size = batch_size or self.expected_batch_size
        for start in range(0, len(lines), size):
            batch = lines[start:start + size]
            line_ids, positions = [], []
            for line in batch:
                self.module.parse_first_line(line)
                ids = native.tokenizer.encode(line, add_special_tokens=False)
                line_ids.append(ids)
                positions.append(self.module.teacher_positions(length, ids))
            keep = sorted({position for row in positions for position in row})
            keep_index = {position: i for i, position in enumerate(keep)}
            total_length = length + max(map(len, line_ids))
            ids = torch.full((len(batch), total_length), native.tokenizer.pad_token_id,
                             dtype=torch.long, device=native.model.device)
            mask = torch.zeros_like(ids)
            for i, continuation in enumerate(line_ids):
                row = prefix + continuation
                ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=ids.device)
                mask[i, :len(row)] = 1
            with torch.inference_mode():
                output = native.model(input_ids=ids, attention_mask=mask, use_cache=False,
                                      logits_to_keep=torch.tensor(keep, device=ids.device))
                for i, pos in enumerate(positions):
                    selected = output.logits[i, [keep_index[p] for p in pos]][:, list(self.module.DIGIT_IDS)]
                    expected, argmax = self.module.expected_from_digit_logits(selected)
                    all_results.append({"expected": expected, "conditional_argmax": argmax})
            del output, ids, mask
        return all_results

    def score(self, prompt, text, *, n=64, seed=15042, progress=None):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        if n < 1:
            raise ValueError("Need at least one accepted line")
        native, module = self.native, self.module
        encoded = native.prepare_input(prompt, text)
        length = int(encoded["input_ids"].shape[1])
        encoded = {key: value.to(native.model.device) for key, value in encoded.items()}
        tokenizer = native.tokenizer

        class FirstNewline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                done = ["\n" in tokenizer.decode(row[length:], skip_special_tokens=True)
                        for row in input_ids]
                return torch.tensor(done, device=input_ids.device, dtype=torch.bool)

        samples, rejected = [], []
        # With scalar teacher forcing, (fixed input, exact line) always denotes
        # the same computation. Preserve every draw and its multiplicity; only
        # memoize this deterministic calculation within this one score() call.
        expected_cache, expected_cache_hits = {}, 0
        attempted = 0
        started = time.perf_counter()
        device_index = native.model.device.index
        with torch.random.fork_rng(devices=[device_index]):
            torch.manual_seed(seed)
            while len(samples) < n and attempted < n * self.max_attempt_multiplier:
                size = min(self.batch_size, n - len(samples), n * self.max_attempt_multiplier - attempted)
                batch = {key: value.repeat(size, 1) for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = native.model.generate(
                        **batch, max_new_tokens=native.max_new_tokens, do_sample=True,
                        temperature=self.temperature, top_k=0, top_p=1.0, min_p=None,
                        typical_p=1.0, num_beams=1, num_return_sequences=1,
                        repetition_penalty=1.0, no_repeat_ngram_size=0,
                        stopping_criteria=StoppingCriteriaList([FirstNewline()]),
                        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                accepted_batch = []
                for row in generated:
                    attempted += 1
                    tokens = row[length:].tolist()
                    completion = tokenizer.decode(tokens, skip_special_tokens=True)
                    line = completion.split("\n", 1)[0].rstrip("\r")
                    item = {"attempt": attempted, "line": line, "completion": completion,
                            "generated_token_ids": tokens}
                    try:
                        item["integers"] = module.parse_first_line(line)
                        module.teacher_positions(length, tokenizer.encode(line, add_special_tokens=False))
                    except (ValueError, module.ScoreParseError) as exc:
                        rejected.append({**item, "reason": str(exc)})
                    else:
                        accepted_batch.append(item)
                del generated, batch
                if accepted_batch:
                    if not self.include_expected:
                        samples.extend(accepted_batch)
                    elif self.expected_batch_size == 1:
                        missing = list(dict.fromkeys(r["line"] for r in accepted_batch if r["line"] not in expected_cache))
                        if missing:
                            calculated = self.expected_for_lines(prompt, text, missing, batch_size=1)
                            expected_cache.update(zip(missing, calculated))
                        expected_cache_hits += len(accepted_batch) - len(missing)
                        expected = [expected_cache[r["line"]] for r in accepted_batch]
                    else:
                        expected = self.expected_for_lines(prompt, text, [r["line"] for r in accepted_batch])
                    if self.include_expected:
                        samples.extend({**sample, **values} for sample, values in zip(accepted_batch, expected))
                if progress:
                    progress({"accepted": len(samples), "requested": n, "attempted": attempted,
                              "rejected": len(rejected), "elapsed_s": time.perf_counter() - started})
        if len(samples) != n:
            error = RuntimeError(f"Only {len(samples)}/{n} valid lines in {attempted} attempts")
            error.partial_samples, error.rejected = samples, rejected
            raise error
        return {"samples": samples, "rejected": rejected, "attempted": attempted,
                "n": n, "seed": seed, "temperature": self.temperature,
                "generation_batch_size": self.batch_size, "expected_batch_size": self.expected_batch_size,
                "conditional_expectations_computed": self.include_expected,
                "deterministic_expected_cache_hits": expected_cache_hits,
                "deterministic_expected_unique_lines": len(expected_cache) if self.expected_batch_size == 1 else None,
                "input_tokens": length, "latency_s": time.perf_counter() - started,
                "sampling": "full_vocabulary_top_k_0_top_p_1_first_N_valid_lines",
                "summary": summarize_samples(samples, [k for k in (8, 16, 32, 64) if k <= n],
                    methods=('integer_mean','expected_mean') if self.include_expected else ('integer_mean',)) if n >= 8 else {}}
