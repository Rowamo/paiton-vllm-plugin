# Paiton on RDNA4: making a 27B INT4 model fast on a 32 GiB Radeon

AMD's Radeon AI PRO R9700 has enough memory for a useful 27B model—but only if
we take quantization seriously. We ported Paiton's previously CDNA-focused
compiler/runtime path to RDNA4, compiled a specialized artifact for AMD's
`Qwen3.8-27B-Quark-Qronos-INT4-W4A16` checkpoint, and then compared it with the
fastest public Linux paths we could make complete without modifying a
competitor.

The result is strongest where interactive inference spends most of its time:
decode. On our controlled `128 input + 128 output` test, Paiton takes 5.985
seconds versus 11.251 seconds for the working public vLLM/ROCm path—1.88x
faster. Paiton is also faster on all four other tested prompt/generation shapes.

Before the graphs, one important clarification: the downloadable Paiton `.so`
is **not a 27B model in a 6.7 MB file**. It is compiled graph and kernel code.
It loads AMD's unchanged 19.9 GB INT4 checkpoint at runtime. Our Hugging Face
release will contain the complete ready-to-serve directory; the matching
GitHub release will contain the small executable overlay, hashes, installer,
and raw evidence.

| Item | Qualified value |
| --- | --- |
| GPU | Radeon AI PRO R9700 (`gfx1201`) |
| Model | `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16` |
| Quantization | Quark Qronos W4A16 INT4, group 128 |
| Paiton artifact | 6,699,696 bytes |
| Artifact SHA256 | `b3b24c9341c2d28b849e06f5842cfb607af82bb208dea49ef72bffabbe933761` |
| Runtime | Ubuntu 24.04, ROCm 7.14, PyTorch 2.12, pinned vLLM |
| Contract | TP1, batch 1, context 8,192, text only |

## Why INT4, not BF16

This is a deliberately practical target. A BF16 27B checkpoint alone is well
beyond the R9700's 32 GiB. AMD's packed Qronos checkpoint is approximately
19.9 GB (18.5 GiB), leaving space for runtime allocations, recurrent state,
KV cache, and the compiled graph. We are not spending engineering time making
a non-quantized model almost fit; Paiton's RDNA work targets models that people
can actually serve on this class of card.

The format is also more specific than the label “INT4” suggests. The checkpoint
stores packed I32 tensors representing signed 4-bit weights, per-group F32
scales, and packed zero points. Activations and KV cache remain BF16. Treating
it as a generic half-byte tensor—or silently as FP8—would give the wrong loader,
the wrong GEMM interface, and potentially plausible but incorrect logits.

## What had to change

RDNA4 support was not an allow-list edit. The old stack assumed CDNA wave64 in
places where `gfx1201` executes wave32. The target model adds another layer of
complexity: 64 text layers alternate three gated-delta-network layers with one
full-attention layer, for 48 recurrent layers and 16 attention layers.

We added or changed five connected parts:

1. **Target and ABI.** Paiton now emits and validates an explicit
   `gfx1201`/wave32/WMMA artifact contract. Artifact selection fails closed on
   GPU architecture, tensor parallelism, shape/layout, binary ABI, size, and
   SHA256 instead of loading whichever `.so` happens to have a matching name.
2. **Quark Qronos loading.** The loader streams the checkpoint, validates all
   496 quantized linears, converts the declared nibble order/sign convention,
   and keeps weights packed. It never materializes a full BF16 copy.
3. **W4A16 execution.** BF16 activations feed packed-I32/F32-group-scale
   operators. Decode and prefill use different RDNA4 paths, including AOT
   Triton kernels embedded into the final `.so`.
4. **Hybrid model state.** Paiton implements the 48 BF16 convolution states,
   48 FP32 recurrent matrices, and KV pages only for the 16 full-attention
   layers. Prefill, token-by-token decode, page reuse, and scheduler state were
   validated against the pinned reference.
5. **Fusion.** The final artifact uses tiled attention, chunk-parallel GDN,
   an AOT causal convolution, and a fused GDN output/gated-RMSNorm epilogue.
   The last fusion is what moved the 1,024-token case from a near tie to a
   clean win over the complete public route.

Every rejected kernel family stayed rejected. For example, two shared-memory
rocWMMA prefill designs were correct but slower, so we replaced the dataflow
rather than tuning the same losing design repeatedly. A BF16-arithmetic
causal-convolution variant changed model token hashes and was removed. The
release log records these branches so they are not rediscovered later.

## A benchmark that is hard to game

We pinned the AMD checkpoint to revision
`649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2` and verified its 19.9 GB file as
SHA256 `32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454`.
Every backend used the same weights, tokenizer, token-ID prompts, TP1, batch 1,
8,192-token context, 2 GiB hybrid-cache reservation, eager execution, and
disabled prefix caching. We used greedy forced output lengths, excluded model
loading, warmed each shape once, synchronized the GPU around measurements, and
ran each backend in a fresh process.

Paiton and the short AITER matrix have five measured samples per workload. The
complete public-compatible result currently has three; we say that explicitly
rather than calling every row a five-sample result. Shared generated-token
hashes match across implementations.

`128 + 128` means 128 input tokens followed by exactly 128 generated tokens.

| Input + output | Paiton median | Public compatible median | Paiton advantage |
| ---: | ---: | ---: | ---: |
| 128 + 1 | 0.13744 s | 0.15749 s | 12.7% |
| 1024 + 1 | 0.64603 s | 0.68018 s | 5.0% |
| 4096 + 1 | 2.91480 s | 3.26702 s | 10.8% |
| 128 + 128 | 5.98515 s | 11.25114 s | 46.8% (1.88x) |
| 4096 + 128 | 13.82578 s | 14.42281 s | 4.1% |

The “public compatible” label is intentional. The working Linux reference is a
vLLM/ROCm stack: W4 linears, GDN, and attention can route through vLLM's RDNA
hybrid kernels, Triton/FLA, ROCm attention, or parts of AITER. Setting one AITER
environment variable does not prove that every operator came from AITER.

## The fairest AITER comparison we could run

AITER is AMD's optimized kernel library, so omitting it would make the result
easy to dismiss. We used the unmodified official `v0.1.19` tag and verified
provider selection. AITER itself labels the R9700 (`gfx1201`) **experimental**
and notes that Triton and many HIP kernels work on RDNA while most CK and
assembly kernels remain CDNA-only.

Its unified-attention path completes three short workloads:

| Input + output | Paiton | AITER unified | Interpretation |
| ---: | ---: | ---: | --- |
| 128 + 1 | 0.13744 s | 0.15370 s | Paiton 10.6% faster |
| 1024 + 1 | 0.64603 s | 0.64722 s | Tie; p10–p90 ranges overlap |
| 128 + 128 | 5.98515 s | 10.92483 s | Paiton 1.83x faster |

At 4,096 tokens, the released unified path selects a Triton kernel requesting
65,792 bytes of LDS; the R9700 limit is 65,536 bytes. We did not patch AITER,
reuse a partially completed timing, or call a crash zero throughput. That
configuration is simply unsupported on the tested stack. Turning unified
attention off produces the complete public-compatible baseline in the first
table.

## What about SGLang?

We looked, because a faster public SGLang route should be the comparison.
AMD's current [ROCm SGLang guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html)
does support Radeon setup, but it warns that default AITER attention may fail
on the R9700, RX 9070 XT, and W7900. AMD recommends Triton attention or
disabling AITER on affected Radeon systems.

More importantly, current upstream SGLang's native Quark dispatcher covers
Quark FP8/MXFP4 schemes, not this native Qronos `W4A16Int4` scheme. The AMD
model card supplies a vLLM command and links the required vLLM changes; it does
not provide an SGLang recipe for this checkpoint. Converting the checkpoint or
implementing a new SGLang loader would no longer be the same public out-of-box
baseline. So there is no SGLang number in the chart—and no claim that Paiton is
faster than an implementation that cannot currently load the model.

For wider context, AMD has published “up to 51.8 tokens/s” for Qwen3.8 27B on
an R9700. That result uses Windows, `llama.cpp`, Vulkan, and MTP=2. It is worth
mentioning, but it is not comparable to a Linux vLLM batch-1 W4A16 run with a
different benchmark contract.

## Serving, not just a microbenchmark

We also started real vLLM servers and ran a 16-request ShareGPT-style smoke
test with client concurrency 8 and forced 128-token outputs. The artifact is
compiled for one sequence, so those requests queue; this measures an API path,
not native eight-sequence batching.

| Server | Completed | Duration | Output throughput |
| --- | ---: | ---: | ---: |
| Paiton | 16/16 | 100.693 s | 20.339 tok/s |
| Public vLLM/ROCm | 16/16 | 185.540 s | 11.038 tok/s |
| AITER enabled + Triton attention | 16/16 | 191.611 s | 10.688 tok/s |

Paiton reaches 1.84x the output-token throughput of the fastest successful
public configuration in this constrained smoke. We are not promoting that as
a production-concurrency headline. The publication plan calls for a repeated
1,024-request run with raw TTFT, TPOT, ITL, end-to-end latency, failures, and
server logs.

## Run it yourself

For normal users, the entire installation is one command:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

We validated the candidate form of this image end to end on the R9700. With a
verified complete model directory mounted in place of the not-yet-published
Hub repository, it loaded the unchanged checkpoint, reported 16.96 GiB of
model memory, reached the health endpoint, and returned
`PAITON_RDNA4_OK` from the OpenAI-compatible chat API. That validates the
container, plugin, `.so`, weights, and server together. We will repeat the
test from an empty cache using only public bytes before calling the release
final.

The container defaults are the validated model, immutable Hugging Face
revision, plugin, ROCm/vLLM runtime, TP1, batch 1, 8,192-token context, and
2 GiB cache reservation. The named volume preserves the approximately 19.9 GB
download. In the final post we will use the image digest rather than only the
human-friendly tag.

The release contains a verifier and an overlay installer. The verifier checks
the executable, generated config, compiled-artifact manifest, GPU/TP/context
contract, and optional 19.9 GB checkpoint hash before vLLM can load the `.so`.
The installer downloads the immutable AMD revision—or accepts an existing
snapshot—and hardlinks the weights where possible so local testing does not
waste another 19.9 GB.

The company publication locations are fixed; immutable tags/digests will be
inserted after the final clean-room gate:

- **Complete Hugging Face model:**
  `EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4@<immutable-tag>`
- **GitHub binary bundle:**
  `https://github.com/Eliovp-BV/paiton-vllm-plugin/releases/tag/<release-tag>`
- **Qualified runtime image:** `<image>@sha256:<digest>`
- **Raw benchmark files:** `<release-results-url>`

The exact install, serve, offline benchmark, and `vllm bench serve` commands
live beside the release. No compiler is required to use the precompiled
artifact.

## Limits we are not hiding

This binary has been validated only on the R9700/`gfx1201` with the pinned ROCm
7.14 stack. It is TP1, batch 1, text only, and capped at 8,192 tokens. It does
not claim AWQ, GLM, MTP/speculative decoding, vision/video, `gfx1200`, another
ROCm release, or current SGLang support. The 16-request serving result is a
smoke test. And because a `.so` is executable code, users should verify its
hash and run it in the pinned container before `dlopen`.

That narrow support contract is a feature, not a loophole: it tells you exactly
what was compiled, exactly where it won, and exactly what remains to be built.
Paiton now gives a 32 GiB RDNA4 card a specialized path for a useful 27B INT4
model—and the biggest current gain is in the decode-heavy workload users feel
most directly.

## Sources

- [AMD Qwen3.8 Qronos model card](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
- [AMD ROCm SGLang guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html)
- [AITER v0.1.19](https://github.com/ROCm/aiter/tree/v0.1.19)
- [vLLM Quark W4A16 PR #48606](https://github.com/vllm-project/vllm/pull/48606)
- [vLLM RDNA/KFD PR #46110](https://github.com/vllm-project/vllm/pull/46110)
- [SGLang Quark dispatcher](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/quark/quark.py)
- [AMD Qwen3.8 Radeon article](https://www.amd.com/en/blogs/2026/run-qwen-3-8-27b-on-amd-ryzen-ai-max-and-radeon-graphics-cards-day-0.html)
