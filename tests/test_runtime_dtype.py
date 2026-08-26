from __future__ import annotations

import torch

from paiton_vllm_plugin.runtime.core.dtype import (
    dtype_str_to_enum,
    dtype_to_enumerator,
    get_dtype_size,
)
from paiton_vllm_plugin.runtime.core.model import torch_dtype_to_string


def test_uint8_matches_native_paiton_dtype_abi():
    assert torch_dtype_to_string(torch.uint8) == "uint8"
    assert get_dtype_size("uint8") == 1
    assert dtype_str_to_enum("uint8") == 9
    assert dtype_to_enumerator("uint8") == "PaitonDtype::kUInt8"
