# Third-party notices — review draft

This notice inventory is prepared for the Paiton Qwen3.8 RDNA4 binary bundle.
It must be reviewed against the final linked binary and approved by the Paiton
rights holder before publication. It is not a replacement for complete license
texts in `LICENSES/`.

## Content or adapted source represented in the artifact

| Component | Upstream | License / notice |
| --- | --- | --- |
| AMD Qwen3.8 Qronos checkpoint | <https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16> | Apache License 2.0 |
| Qwen3.8 base model | <https://huggingface.co/Qwen/Qwen3.8-27B> | Apache License 2.0 |
| AITemplate-generated host/runtime components | <https://github.com/facebookincubator/AITemplate> | Apache License 2.0; Meta Platforms notices |
| vLLM paged-attention/runtime code | <https://github.com/vllm-project/vllm> at `39bd959b582c85e78e7e0326d49042ce7c3c07ed` | Apache License 2.0; vLLM contributors |
| Qwen GDN Triton stages adapted through vLLM | <https://github.com/fla-org/flash-linear-attention> | MIT; Copyright 2023–2025 Songlin Yang, Yu Zhang |
| RDNA4 causal-convolution stage adapted from AITER | <https://github.com/ROCm/aiter> | MIT; Copyright 2024–2026 Advanced Micro Devices, Inc. |
| Triton AOT-generated host/device code | <https://github.com/ROCm/triton> | MIT; retain applicable upstream generator notices |

## Build and runtime dependencies

The bundle dynamically links ROCm libraries including `libamdhip64`,
`librocblas`, `libhipblaslt`, `libhipblas`, `librocrand`, and `librccl`. These
libraries are not embedded in the small GitHub bundle; the qualified runtime
container supplies them. Their versions, licenses, and relationships must be
represented in the final SBOM.

Composable Kernel revision `3df477638d05169dbe73f4d58e3a13ad3ca7b5da` is
part of the compiler toolchain. The final binary audit must distinguish code
actually included in the artifact from a build-only dependency before the
notice is finalized.

## Required final bundle material

Before release, include at least:

- the Paiton project license selected by the rights holder;
- the complete Apache License 2.0 text;
- the complete MIT license text and retained copyright notices above;
- any other license discovered by the final generated-source/SBOM audit;
- this notice file without the “review draft” qualification;
- `MODIFICATIONS.md`, identifying changes to the upstream model package.

No repository-level license currently exists in either Paiton repository.
Package metadata saying `Apache-2.0` is not by itself a complete public license
grant. Do not distribute the executable until that is resolved.
