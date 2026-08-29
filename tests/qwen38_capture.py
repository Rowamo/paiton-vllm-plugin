"""Stable result serialization shared by Qwen3.8 hardware gates."""

from __future__ import annotations

import math
from typing import Any


def capture_completion(output: Any, prompt: list[int]) -> dict[str, Any]:
    completion = output.outputs[0]
    steps: list[list[dict[str, Any]]] = []
    for candidates in completion.logprobs or ():
        ranked = sorted(
            candidates.items(),
            key=lambda item: (
                item[1].rank if item[1].rank is not None else 1 << 30,
                item[0],
            ),
        )
        steps.append(
            [
                {
                    "token_id": int(token_id),
                    "logprob": float(logprob.logprob),
                    "rank": (
                        int(logprob.rank) if logprob.rank is not None else None
                    ),
                }
                for token_id, logprob in ranked
            ]
        )
    return {
        "prompt_token_ids": list(prompt),
        "generated_token_ids": list(completion.token_ids),
        "finish_reason": completion.finish_reason,
        "top_logprobs": steps,
    }


def assert_finite_completion(result: dict[str, Any]) -> None:
    if not result["top_logprobs"]:
        raise AssertionError("completion did not expose logprobs")
    for step, candidates in enumerate(result["top_logprobs"]):
        if not candidates:
            raise AssertionError(f"completion step {step} has no candidates")
        for candidate in candidates:
            if not math.isfinite(candidate["logprob"]):
                raise AssertionError(
                    f"completion step {step} token {candidate['token_id']} "
                    f"has non-finite logprob {candidate['logprob']}"
                )


def assert_reference_completion(
    actual: dict[str, Any],
    reference: dict[str, Any],
    *,
    winner_logprob_atol: float = 0.5,
    minimum_top20_overlap: int = 10,
) -> None:
    assert_finite_completion(actual)
    assert_finite_completion(reference)
    if actual["prompt_token_ids"] != reference["prompt_token_ids"]:
        raise AssertionError("reference prompt does not match the tested prompt")
    if actual["generated_token_ids"] != reference["generated_token_ids"]:
        raise AssertionError(
            "generated tokens differ: "
            f"actual={actual['generated_token_ids']} "
            f"reference={reference['generated_token_ids']}"
        )
    if len(actual["top_logprobs"]) != len(reference["top_logprobs"]):
        raise AssertionError("completion step count differs from reference")
    for step, (actual_candidates, reference_candidates) in enumerate(
        zip(actual["top_logprobs"], reference["top_logprobs"], strict=True)
    ):
        actual_winner = actual_candidates[0]
        reference_winner = reference_candidates[0]
        if actual_winner["token_id"] != reference_winner["token_id"]:
            raise AssertionError(
                f"step {step} winner differs: actual={actual_winner['token_id']} "
                f"reference={reference_winner['token_id']}"
            )
        winner_delta = abs(actual_winner["logprob"] - reference_winner["logprob"])
        if winner_delta > winner_logprob_atol:
            raise AssertionError(
                f"step {step} winner logprob delta {winner_delta} exceeds "
                f"{winner_logprob_atol}"
            )
        actual_ids = {candidate["token_id"] for candidate in actual_candidates[:20]}
        reference_ids = {
            candidate["token_id"] for candidate in reference_candidates[:20]
        }
        overlap = len(actual_ids & reference_ids)
        if overlap < minimum_top20_overlap:
            raise AssertionError(
                f"step {step} top-20 overlap {overlap} is below "
                f"{minimum_top20_overlap}"
            )
