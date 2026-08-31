# Paiton vLLM Plugin

Paiton runs architecture-specific compiled models through a vLLM-compatible
API on AMD GPUs.

## Qwen3.8 INT4 on RDNA4

The first public release targets AMD's
[`Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
checkpoint on the Radeon AI PRO R9700 (`gfx1201`). It uses the unchanged AMD
INT4 weights with a 6.7 MB Paiton `.so` containing the compiled graph and
custom RDNA4 kernels. The `.so` is executable code, not another copy of the
model weights.

After the public image and model tag are published, start the complete server
with one command:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

The first run downloads the public
[`EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4`](https://huggingface.co/EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4)
repository into the named volume; later runs reuse it.

Already have AMD's exact checkpoint in the normal Hugging Face cache? Reuse it
without downloading or copying the 19.9 GB weights:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 --mount "type=bind,src=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub},dst=/models/base-cache,readonly" -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

That mount is read-only and excludes the user's token. Paiton links the cached
checkpoint in place and downloads only the approximately 7.5 MB overlay into
the named volume; it cannot write root-owned files into the host cache.

An unpacked checkpoint outside the Hub cache is also reusable without copying:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -e PAITON_BASE_MODEL=/models/base -v /absolute/path/to/amd-qwen38:/models/base:ro -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

The whole-cache command lets the executable container read every blob in that
mounted cache. Users who also cache private/gated models should use the explicit
base-directory command and mount only AMD's exact snapshot.

The image already contains the tested ROCm/vLLM runtime and this plugin, pins
the model revision, validates the target and `.so` checksum before loading it,
and starts the OpenAI-compatible API as model `qwen38`. Users do not need the
private Paiton compiler or a local Python installation.

Check the running server:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen38",
    "messages": [{"role": "user", "content": "Reply with exactly: PAITON_RDNA4_OK"}],
    "temperature": 0,
    "max_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

For supply-chain reproducibility, the final release notes also provide the
same image as an immutable `@sha256:<digest>` reference.

## Validated contract

| Component | Qualified value |
| --- | --- |
| GPU | Radeon AI PRO R9700, `gfx1201` |
| Host runtime | Ubuntu 24.04, ROCm 7.14 |
| Model | Quark Qronos W4A16 INT4, group 128 |
| Activations and KV cache | BF16 |
| Tensor parallelism / batch | TP1 / batch 1 |
| Maximum model length | 8,192 tokens |
| Scope | Text only; no vision, MTP, or speculative decoding |

This exact binary is not a generic promise for other RDNA4 devices, ROCm
versions, quantization formats, tensor-parallel sizes, or batch sizes.

## Release and benchmark details

The complete technical guide, hashes, audit installer, public-baseline and
AITER commands, raw benchmark results, and current publication gates are in
[`release/qwen38-qronos-gfx1201/README.md`](release/qwen38-qronos-gfx1201/README.md).
The readable engineering report is
[`docs/qwen38-rdna4-blog.md`](docs/qwen38-rdna4-blog.md).

The release remains an internal candidate until the reviewed company source
commit, Hugging Face model, approved notices, source/model tags, and immutable
container digest are published. Do not treat the friendly image tag above as
available before that announcement.
