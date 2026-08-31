# Third-party notices

This notice inventory accompanies the Paiton Qwen3.8 RDNA4 source, binary
bundle, runtime image, and Hugging Face model package. The corresponding
license texts are included in `LICENSES/`.

## Content or adapted source represented in the release

| Component | Component/source | License / retained notice |
| --- | --- | --- |
| AMD Qwen3.8 Qronos checkpoint | `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16@649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2` | Apache License 2.0 |
| Qwen3.8 base model | `Qwen/Qwen3.8-27B` | Apache License 2.0 |
| AITemplate-generated host/runtime components | `facebookincubator/AITemplate` | Apache License 2.0; Meta Platforms notices |
| vLLM paged-attention/runtime code | `vllm-project/vllm@39bd959b582c85e78e7e0326d49042ce7c3c07ed` | Apache License 2.0; vLLM contributors |
| Qwen GDN stages adapted through vLLM/FLA | `fla-org/flash-linear-attention` | MIT; Copyright 2023–2025, Songlin Yang, Yu Zhang |
| RDNA4 causal-convolution stage adapted from AITER | `ROCm/aiter@v0.1.19` | MIT; Copyright Advanced Micro Devices, Inc. |
| Triton AOT-generated host/device code | `ROCm/triton` | MIT; Copyright 2018–2020 Philippe Tillet; Copyright 2020–2022 OpenAI |

## Build/toolchain input

The compiler was configured with Composable Kernel at
`ROCm/composable_kernel@3df477638d05169dbe73f4d58e3a13ad3ca7b5da`.
It is recorded as a build dependency; this is not a claim that CK code is
embedded in the compiled artifact. Its MIT license and notices are retained.

## Retained source notices

The compiled/generated source retains the applicable upstream headers,
including:

- `Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.`
  on the AMD causal-convolution kernel lineage;
- `Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.`
  on AMD tensor utility code;
- `Copyright (c) 2024, The vLLM team.` and vLLM contributor notices on
  paged-attention and Qwen GDN-derived code;
- `Copyright (c) 2023-2025, Songlin Yang, Yu Zhang` on the FLA-derived GDN
  stages; and
- `Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.` on
  the accepted generated HIP source.

The Paiton project and release overlay are distributed under Apache License
2.0. `MODIFICATIONS.md` describes changes made around the unchanged AMD model
files. The complete Apache License text is available as `LICENSE` and
`LICENSES/Apache-2.0.txt`. Applicable MIT license texts and copyright notices
are included separately.

## Runtime dependencies

The `.so` dynamically links ROCm libraries including `libamdhip64`,
`librocblas`, `libhipblaslt`, `libhipblas`, `librocrand`, and `librccl`. These
libraries are not embedded in the small GitHub bundle; the qualified runtime
container supplies them. The compiled-artifact SBOM distinguishes included
code, build-time inputs, and direct runtime relationships. The final image will
be paired with its own generated dependency-level SBOM.
