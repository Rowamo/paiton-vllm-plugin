# Benchmark evidence

These JSON files are the raw outputs used for the release candidate. They are
kept unchanged, including local source paths, so readers can audit the actual
sample arrays and environment capture.

## Controlled offline runs

| File | Backend | Samples | Accepted scope |
| --- | --- | ---: | --- |
| `offline/paiton-five-sample.json` | Final Paiton artifact | 5 per workload | Full five-workload matrix |
| `offline/public-rocm-three-sample.json` | Compatible public vLLM/ROCm path | 3 per workload | Full five-workload matrix |
| `offline/aiter-unified-short-five-sample.json` | Released AITER v0.1.19 unified attention | 5 per workload | Three successful short workloads only |

The Paiton and public-compatible sample counts differ. Tables must say so; do
not describe the public-compatible file as a five-sample run. Shared
output-token hashes match.

## Online serving smoke

The four serving files contain a single 16-prompt ShareGPT-style run per
configuration. Paiton, public ROCm, and AITER with Triton attention completed
16/16. The AITER unified run completed only 7/16 before the engine died and is
failure evidence, not a valid performance result.

The server artifact was compiled for batch 1. Client concurrency 8 therefore
created queued requests, not true eight-sequence execution. These files are a
serving smoke test and must not be described as a final production-capacity
benchmark.

## One-line container smoke

`one-line-container-smoke.json` records the successful R9700 test of the
release image entrypoint against a verified complete model directory. The
container reached `/health`, loaded 16.96 GiB of model data, and returned
`PAITON_RDNA4_OK` from `/v1/chat/completions`.

This proves the image, plugin, `.so`, unchanged checkpoint, and API work
together. It is not yet the final clean-room publication test: the candidate
used a local mounted model because the public `EliovpAI` model revision and
GHCR image do not exist yet. The final gate repeats the same command from an
empty cache using only published bytes.
