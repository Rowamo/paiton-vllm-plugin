# Paiton Qwen3.8 Qronos W4A16 for RDNA4

This release candidate adds a Paiton-compiled execution path for
[`amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16`](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
on the AMD Radeon AI PRO R9700 (`gfx1201`). It is deliberately an INT4 release:
the unchanged AMD checkpoint is 19,893,384,832 bytes and fits a 32 GiB card;
this work does not target the non-quantized checkpoint.

The Paiton `.so` is **not the model weights**. It is 6,699,696 bytes of compiled
host code, graph logic, and RDNA4 kernels. At runtime it loads AMD's unchanged
19.9 GB Quark Qronos W4A16 checkpoint. A runnable model directory therefore
contains both the AMD files and the Paiton overlay.

## The simple path: one command

After publication, an R9700 user with the AMD driver and Docker runs:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

That is the normal installation. The image already pins the Paiton plugin,
ROCm/vLLM runtime, company Hugging Face model, immutable model revision, TP1,
batch 1, 8,192-token context, 2 GiB cache reservation, and API port. The first
run downloads the complete public package into the named volume; subsequent
runs reuse it.

If the exact AMD snapshot is already in the user's normal Hub cache, this
equally direct command avoids downloading or copying the weights:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 --mount "type=bind,src=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub},dst=/models/base-cache,readonly" -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

The bind exposes the host cache read-only, not the Hugging Face token. Paiton
links the 19.9 GB checkpoint in place and downloads only the approximately 7.5
MB overlay into the named volume; it cannot write into the host cache.

An existing unpacked checkpoint is equally direct:

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc=host -p 8000:8000 -e PAITON_BASE_MODEL=/models/base -v /absolute/path/to/amd-qwen38:/models/base:ro -v paiton-qwen38-cache:/models/cache ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1
```

That path creates a small overlay in the named Docker cache while keeping the
mounted checkpoint read-only. Users who prefer a completely isolated download
can omit both reuse options and mount only
`-v paiton-qwen38-cache:/models/cache`.

The whole-cache command lets the executable container read every blob in that
mounted cache even though it cannot modify it. Users who also cache
private/gated models should bind only AMD's exact snapshot with
`PAITON_BASE_MODEL`, as shown above.

Reuse verifies the full checkpoint SHA256 by default. Setting
`PAITON_VERIFY_BASE_SHA256=0` skips that integrity gate and is unsafe for a
published benchmark or reproducible result; it exists only for a user who
explicitly accepts a size-only check of their own trusted local file.

For a cryptographically reproducible run, the final release replaces the image
tag with its published `@sha256:<digest>`. The longer procedures below exist
for audit, offline installation, and independent benchmarking—not because
ordinary users should have to assemble the model themselves.

## Validated contract

| Component | Qualified value |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 |
| GPU ISA | `gfx1201`, RDNA4, wave32/WMMA |
| Host | x86-64 Ubuntu 24.04 |
| ROCm / HIP | `7.14.60850` |
| PyTorch | `2.12.0+rocm7.14.0` |
| vLLM | `39bd959b582c85e78e7e0326d49042ce7c3c07ed` |
| Source checkpoint | `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16` |
| Source revision | `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2` |
| Quantization | Quark Qronos, packed INT4 weights, group 128, BF16 activations |
| Tensor parallelism | 1 |
| Maximum batch | 1 sequence |
| Maximum model length | 8,192 tokens |
| Maximum batched tokens | 8,192 |
| Scope | Text only; no vision/video or speculative decoding |

This is not a claim of generic Linux, generic RDNA4, `gfx1200`, another ROCm
release, or another Qwen quantization working with the same binary. The shared
library has minimum host ABI requirements including GLIBC 2.38, GLIBCXX
3.4.30, and CXXABI 1.3.13.

## Exact release bytes

| File | Size | SHA256 |
| --- | ---: | --- |
| Compiled `.so` | 6,699,696 | `b3b24c9341c2d28b849e06f5842cfb607af82bb208dea49ef72bffabbe933761` |
| Artifact manifest | 789,856 | `174fc432a779a8fade9ce97082ccecef4a019f0202f02e1db164b6b3bb6c901b` |
| Generated config | 13,647 | `1974563798648387a8c63130906dd08b1451083e7f273c14116840637511fb85` |
| AMD `model.safetensors` | 19,893,384,832 | `32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454` |

Always verify the executable before vLLM loads it. A repository host's malware
scan is useful, but it is not a substitute for checking the pinned revision and
SHA256.

The included in-toto/SLSA-shaped provenance is an unsigned, self-reported
record of the private compiler inputs. It is not a third-party attestation and
does not claim a publicly reproducible binary build. The final OCI image will
be paired with a separately generated dependency SBOM.

The `urn:eliovp-bv:paiton:*` builder and build-type values are stable,
release-local identifiers, not resolvable public services. The provenance
explicitly marks its dependency inventory incomplete because the private build
did not record immutable revisions for every transitive source generator; known
compiler, plugin, vLLM, model, CK, Triton, ROCm/HIP, and AITER inputs are listed.

## Publisher-only assembly gate

The tracked manifest intentionally remains a candidate: a Git commit cannot
contain its own hash. After rights approval, commit and tag the clean source,
then let the bundle builder inject that already-known commit and the real UTC
packaging time into generated release assets:

```bash
export PAITON_RELEASE_ID='paiton-qwen38-qronos-w4a16-gfx1201-v1'
export PAITON_SOURCE_REVISION="$(git rev-parse HEAD)"
export PAITON_PACKAGED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
test -z "$(git status --porcelain --untracked-files=all)"
test "$(git rev-parse "refs/tags/$PAITON_RELEASE_ID^{commit}")" = "$PAITON_SOURCE_REVISION"

python3 release/qwen38-qronos-gfx1201/build_bundle.py \
  --artifact-dir /absolute/path/to/verified/compiler-output \
  --output-dir "/absolute/staging/$PAITON_RELEASE_ID" \
  --release-source-revision "$PAITON_SOURCE_REVISION" \
  --packaged-at "$PAITON_PACKAGED_AT" \
  --archive
```

The command refuses a dirty checkout, tag mismatch, candidate manifest,
noncanonical archive root, altered license/metadata bytes, or invalid
SPDX/in-toto subjects. It stages atomically and produces one canonical tar
root. Build the complete Hub tree offline from the generated released manifest:

```bash
python3 release/qwen38-qronos-gfx1201/build_hf_repo.py \
  --overlay-dir /absolute/path/to/verified/complete-overlay \
  --upstream-metadata-dir /absolute/path/to/pinned-amd-metadata \
  --manifest "/absolute/staging/$PAITON_RELEASE_ID/bundle-manifest.json" \
  --output-dir /absolute/staging/paiton-qwen38-hf \
  --checkpoint-mode copy

cd /absolute/staging/paiton-qwen38-hf
sha256sum -c SHA256SUMS
```

`copy` intentionally freezes the publication tree. A hardlink is permitted
only when both source and staged tree are kept immutable/read-only and the
complete checksum file is verified immediately before upload.

After uploading and verifying the Hub tree, capture its immutable 40-character
commit and build the image from a `git archive` of the exact source tag—not the
publisher's working directory:

```bash
export PAITON_HF_COMMIT='<verified-40-character-hub-commit>'
export PAITON_IMAGE='ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1'
export PAITON_BUILD_CONTEXT="$(mktemp -d)"
git archive "$PAITON_RELEASE_ID" | tar -x -C "$PAITON_BUILD_CONTEXT"

docker buildx build \
  --platform linux/amd64 \
  --build-arg PLUGIN_REVISION="$PAITON_SOURCE_REVISION" \
  --build-arg MODEL_REVISION="$PAITON_HF_COMMIT" \
  --provenance=false \
  --sbom=true \
  --file "$PAITON_BUILD_CONTEXT/Dockerfile.qwen38-rdna4" \
  --tag "$PAITON_IMAGE" \
  --push \
  "$PAITON_BUILD_CONTEXT"
```

The source archive prevents dirty or untracked working-tree files from entering
the image. The image SBOM is attached separately; the private/local qualified
base is deliberately not represented by a misleading automatic BuildKit
provenance claim.

## Distribution layout

The final publication uses two matching channels:

1. A GitHub Release contains the small Paiton bundle: `.so`, compiled-artifact
   manifest, generated config, installer, verifier, checksums, notices, and raw
   results. It does not duplicate the 19.9 GB weights.
2. A Hugging Face model repository contains a complete, directly servable
   directory: AMD's unchanged weights and tokenizer plus the exact same Paiton
   `.so`, manifest, and config.

An artifact-only Hugging Face repository cannot currently be passed directly
to `vllm serve`, because the plugin resolves the checkpoint from the same model
reference. Use the complete repository or build a local overlay with the
installer below.

## Option A: complete Hugging Face model

The complete public model will live at
[`EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4`](https://huggingface.co/EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4).
After publication, set the immutable repository revision supplied in the
release notes:

```bash
export PAITON_HF_REPO='EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4'
export PAITON_HF_REVISION='<signed-tag-or-commit>'
export PAITON_MODEL_DIR="$PWD/Qwen3.8-Paiton-RDNA4"

hf download "$PAITON_HF_REPO" \
  --revision "$PAITON_HF_REVISION" \
  --local-dir "$PAITON_MODEL_DIR"

python3 "$PAITON_MODEL_DIR/verify_release.py" \
  --bundle-dir "$PAITON_MODEL_DIR" \
  --model-dir "$PAITON_MODEL_DIR" \
  --verify-checkpoint-sha256
```

Do not substitute `main` for the immutable revision in a reproducible run.

## Option B: GitHub bundle over the AMD checkpoint

Download and extract the asset from the planned company repository,
[`Eliovp-BV/paiton-vllm-plugin`](https://github.com/Eliovp-BV/paiton-vllm-plugin),
then create a local model directory. The default `auto` mode hardlinks files on
the same filesystem and otherwise symlinks them, so it does not copy another
19.9 GB checkpoint.

```bash
export PAITON_BUNDLE="$PWD/paiton-qwen38-qronos-w4a16-gfx1201-v1"
export PAITON_MODEL_DIR="$PWD/Qwen3.8-Paiton-RDNA4"

python3 "$PAITON_BUNDLE/verify_release.py" \
  --bundle-dir "$PAITON_BUNDLE"

python3 "$PAITON_BUNDLE/install_overlay.py" \
  --bundle-dir "$PAITON_BUNDLE" \
  --download \
  --output-dir "$PAITON_MODEL_DIR" \
  --link-mode auto \
  --verify-checkpoint-sha256
```

To reuse an already downloaded exact snapshot:

```bash
python3 "$PAITON_BUNDLE/install_overlay.py" \
  --bundle-dir "$PAITON_BUNDLE" \
  --base-model-dir /models/amd-qwen38-qronos \
  --output-dir "$PAITON_MODEL_DIR" \
  --link-mode auto
```

The full checkpoint hash is optional because reading 19.9 GB takes time; the
installer always checks its exact byte size and the Quark INT4 group-128 config.
Use the hash gate for publication, first installation, and benchmark evidence.

## Runtime

The validated route is the immutable Paiton runtime container that will be
linked by digest in the final release. Until that digest is published, the
candidate is not a portable binary promise. Local installations must exactly
match the runtime table above and install the Paiton plugin from the release
tag.

For a qualified local environment:

```bash
export VLLM_USE_PAITON_PLATFORM=1
export VLLM_DISABLE_PAITON_PLATFORM=0
export PAITON_GPU_ARCH=gfx1201
export HIP_VISIBLE_DEVICES=0

vllm serve "$PAITON_MODEL_DIR" \
  --served-model-name qwen38 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --kv-cache-dtype auto \
  --enforce-eager \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 \
  --port 8000
```

The artifact is batch-1. Raising `--max-num-seqs` does not create a valid
higher-batch artifact and must be rejected by the runtime contract.

### Smoke test

```bash
curl --fail http://127.0.0.1:8000/v1/models

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

The release-candidate container path was exercised end to end on the R9700 on
2026-08-31 using a verified mounted copy of the complete model directory. It
reported 16.96 GiB of model memory, reserved 2 GiB for cache, returned HTTP 200
from `/health`, and produced `PAITON_RDNA4_OK` with `finish_reason=stop`. See
[`results/one-line-container-smoke.json`](results/one-line-container-smoke.json).
The subsequent slim-image gate repeated the full load and API request after
reducing the compressed container to 5.14 GB; see
[`results/slim-container-smoke.json`](results/slim-container-smoke.json).
The final clean-room gate will repeat this from the published image and an
empty cache volume; it cannot be claimed until those public objects exist.

## Controlled offline benchmark

The checked-in harness uses token-ID prompts, greedy forced output lengths,
one warmup per workload, synchronized timing, no detokenization, no prefix
caching, a 2 GiB hybrid-cache reservation, and a fresh process for each
backend. Model loading is reported but excluded.

```bash
export PAITON_PLUGIN=/path/to/paiton-vllm-plugin
export CHECKPOINT_SHA=32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454

VLLM_USE_PAITON_PLATFORM=1 \
PAITON_GPU_ARCH=gfx1201 \
HIP_VISIBLE_DEVICES=0 \
python3 "$PAITON_PLUGIN/benchmarks/benchmark_qwen38_end_to_end.py" \
  --model "$PAITON_MODEL_DIR" \
  --backend-label paiton-release \
  --samples 5 \
  --warmups 1 \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --revision 649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2 \
  --output paiton-release.json
```

The full benchmark matrix is `128x1`, `1024x1`, `4096x1`, `128x128`, and
`4096x128`, where `INPUTxOUTPUT` denotes input and forced output token counts.

## Online ShareGPT serving benchmark

Use the vLLM CLI shipped by the pinned revision. Start the server first, wait
for `/health`, and run the client from a second terminal:

```bash
vllm bench serve \
  --backend vllm \
  --host 127.0.0.1 \
  --port 8000 \
  --endpoint /generate \
  --model qwen38 \
  --dataset-name sharegpt \
  --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1024 \
  --request-rate inf \
  --max-concurrency 8 \
  --sharegpt-output-len 256 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --seed 0 \
  --save-result \
  --result-dir results/serving \
  --result-filename paiton-sharegpt-1024.json
```

Report completed and failed requests, duration, request/output/total-token
throughput, TTFT, TPOT, ITL, end-to-end latency, the exact server command, and
the server log. Since this artifact executes one sequence at a time,
`--max-concurrency 8` measures queued production-style requests, not native
eight-sequence batching.

## Fair public baselines

The primary baseline is the fastest public vLLM/ROCm configuration that loads
the same immutable checkpoint and completes the complete matrix. It is not
accurate to label that whole route “AITER”: its W4 linears, GDN stages, and
attention may come from different vLLM, Triton/FLA, ROCm, and AITER providers.
Keep each provider-selection log with the raw result.

For AITER, use the unmodified official `v0.1.19` tag and its documented install
path:

```bash
git clone --recursive --branch v0.1.19 \
  https://github.com/ROCm/aiter.git /opt/aiter-v0.1.19
cd /opt/aiter-v0.1.19
AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop
```

Then enable it before vLLM imports:

```bash
export VLLM_PAITON_VANILLA_ROCM_PLATFORM=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
```

Do not patch AITER to make a benchmark complete. On the qualified stack its
unified path completes the short matrix but the 4,096-token case selects a
Triton kernel requiring 65,792 bytes of LDS on hardware with a 65,536-byte
limit. Record that configuration as unsupported, not as zero throughput and
not as a partial performance result. The compatible complete baseline instead
sets `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=0` and uses `ROCM_ATTN`.

## Results and limitations

The raw results are under [`results/`](results/README.md). Headline controlled
medians are:

| Input + output | Paiton | Public compatible | Improvement |
| ---: | ---: | ---: | ---: |
| 128 + 1 | 0.13744 s | 0.15749 s | 12.7% faster |
| 1024 + 1 | 0.64603 s | 0.68018 s | 5.0% faster |
| 4096 + 1 | 2.91480 s | 3.26702 s | 10.8% faster |
| 128 + 128 | 5.98515 s | 11.25114 s | 46.8% faster (1.88x) |
| 4096 + 128 | 13.82578 s | 14.42281 s | 4.1% faster |

Paiton used five samples; the accepted complete public result currently
contains three samples. The compared output-token hashes match. AITER's three
successful five-sample short cases show Paiton 10.6% faster at `128+1`, a tie
at `1024+1` (0.18% median difference with overlapping p10-p90 ranges), and
1.83x faster at `128+128`.

The 16-request ShareGPT run is only a preliminary serving smoke test. Paiton
completed 16/16 requests at 20.339 output tokens/s versus 11.038 for the
fastest successful public configuration, but these numbers are not presented
as a final high-concurrency capacity result.

## Ecosystem context

- [AMD's model card](https://huggingface.co/amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16)
  documents vLLM and says the checkpoint needs a Quark-compatible
  `W4A16Int4` runtime, linking two still-unmerged vLLM changes.
- [AITER v0.1.19](https://github.com/ROCm/aiter/tree/v0.1.19) labels the R9700
  `gfx1201` target experimental and notes that many CK/assembly kernels remain
  CDNA-only.
- [AMD's SGLang guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/sglang.html)
  warns that default AITER attention may fail on the R9700 and recommends
  Triton attention or disabling AITER on affected Radeon systems.
- Current [SGLang Quark dispatch](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/quantization/quark/quark.py)
  recognizes Quark FP8/MXFP4 paths, not this native Quark W4A16 scheme. There
  is no honest apples-to-apples SGLang number for this checkpoint without
  implementing or converting it.
- [AMD's Qwen3.8 Radeon article](https://www.amd.com/en/blogs/2026/run-qwen-3-8-27b-on-amd-ryzen-ai-max-and-radeon-graphics-cards-day-0.html)
  reports up to 51.8 tokens/s on an R9700, but that result uses Windows,
  `llama.cpp`, Vulkan, and MTP=2. It is useful ecosystem context, not a directly
  comparable Linux vLLM batch-1 measurement.

## Publication status

This directory is a technically verified release candidate, not yet an
approved public binary release. The company plugin repository now has an
Apache-2.0 root license, but publication remains gated on explicit generated-
artifact and contributor-rights confirmation, third-party notice approval, and
the final immutable GitHub/Hugging Face/GHCR publication gates. See
[`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md). Do not upload the `.so`
until those gates are complete.
