---
base_model: amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16
license: other
license_name: apache-2.0-and-mit
license_link: https://github.com/Eliovp-BV/paiton-vllm-plugin/blob/paiton-qwen38-qronos-w4a16-gfx1201-v1/release/qwen38-qronos-gfx1201/THIRD_PARTY_NOTICES.md
library_name: paiton-vllm-plugin
pipeline_tag: text-generation
tags:
  - paiton
  - vllm
  - rocm
  - rdna4
  - gfx1201
  - r9700
  - qronos
  - quark
  - w4a16
  - int4
---

# Qwen3.8 27B Qronos INT4 for Paiton on RDNA4

This repository is a directly servable Paiton package for AMD's unchanged
[`Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
checkpoint. It combines those original INT4 weights and tokenizer files with
a Paiton-generated config, artifact manifest, and 6.7 MB `gfx1201` `.so`.

The `.so` is executable graph and kernel code, not model weights. Verify its
checksum and use only the qualified runtime image. This package is text-only;
it does not expose the checkpoint's vision/video or MTP paths.

## Run

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

The API serves the model as `qwen38` on port 8000. If AMD's exact source
snapshot is already in the normal Hub cache, the release notes provide a second
one-line command that mounts it read-only and downloads only the approximately
7.5 MB Paiton overlay; weights are not copied or downloaded again. They also
show how to bind an unpacked model directory and use the immutable image digest.

Existing Hugging Face cache, with no weight redownload:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 --mount "type=bind,src=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub},dst=/models/base-cache,readonly" -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

## Qualified contract

| Component | Value |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 (`gfx1201`) |
| Host/runtime | x86-64 Ubuntu 24.04, ROCm 7.14.60850 |
| PyTorch / vLLM | 2.12.0+rocm7.14.0 / `39bd959b582c85e78e7e0326d49042ce7c3c07ed` |
| Quantization | Quark Qronos packed INT4, group 128, BF16 activations |
| Parallelism / batch | TP1 / batch 1 |
| Maximum model length | 8,192 tokens |
| Scope | Text only; no multimodal execution or speculative decoding |

This binary is not a generic promise for another RDNA4 GPU, ROCm version,
quantization format, tensor-parallel size, or batch size.

## Integrity

| File | SHA256 |
| --- | --- |
| Paiton `.so` | `b3b24c9341c2d28b849e06f5842cfb607af82bb208dea49ef72bffabbe933761` |
| Artifact manifest | `174fc432a779a8fade9ce97082ccecef4a019f0202f02e1db164b6b3bb6c901b` |
| Generated config | `1974563798648387a8c63130906dd08b1451083e7f273c14116840637511fb85` |
| Unchanged `model.safetensors` | `32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454` |

Source model revision:
`649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2`.

See the matching
[`Eliovp-BV/paiton-vllm-plugin`](https://github.com/Eliovp-BV/paiton-vllm-plugin)
release for verifier tooling, source, notices, reproducible commands, raw
benchmark evidence, and the exact runtime image digest.

## License and provenance

AMD's source checkpoint, the Qwen base model, and the Paiton-owned overlay are
Apache-2.0. Adapted executable portions retain compatible MIT obligations, so
the repository metadata uses `Apache-2.0 AND MIT` rather than describing the
entire package as Apache-only. The weights are redistributed unchanged;
`MODIFICATIONS.md` describes the runtime overlay. `THIRD_PARTY_NOTICES.md` and
`LICENSES/` map each component to its license and retained notices.

`provenance.intoto.jsonl` is an unsigned, self-reported record of the private
build inputs. It is not a third-party attestation or a claim that the `.so` can
be reproduced from the public repository; the release does provide
reproducible serving inputs by immutable hashes.
