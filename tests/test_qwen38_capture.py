import math
import unittest

from tests.qwen38_capture import (
    assert_finite_completion,
    assert_reference_completion,
)


def _result(tokens=(7, 8), first=-1.0, second=-2.0):
    return {
        "prompt_token_ids": [1, 2],
        "generated_token_ids": list(tokens),
        "finish_reason": "length",
        "top_logprobs": [
            [
                {"token_id": 7, "logprob": first, "rank": 1},
                *[
                    {"token_id": token, "logprob": -3.0 - token, "rank": token}
                    for token in range(2, 21)
                ],
            ],
            [
                {"token_id": 8, "logprob": second, "rank": 1},
                *[
                    {"token_id": token, "logprob": -3.0 - token, "rank": token}
                    for token in range(2, 21)
                ],
            ],
        ],
    }


class Qwen38CaptureTests(unittest.TestCase):
    def test_reference_acceptance_is_fail_closed(self):
        reference = _result()
        assert_reference_completion(_result(first=-1.2, second=-2.1), reference)
        with self.assertRaisesRegex(AssertionError, "non-finite"):
            assert_finite_completion(_result(first=math.nan))
        with self.assertRaisesRegex(AssertionError, "generated tokens differ"):
            assert_reference_completion(_result(tokens=(9, 8)), reference)
        with self.assertRaisesRegex(AssertionError, "logprob delta"):
            assert_reference_completion(_result(first=-2.0), reference)


if __name__ == "__main__":
    unittest.main()
