# Modifications and provenance

This package combines an unchanged upstream checkpoint with a Paiton-specific
runtime overlay.

## Unchanged upstream content

- Base model: `amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16`
- Immutable revision: `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2`
- `model.safetensors` SHA256:
  `32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454`
- The quantized weights, tokenizer vocabulary, merges, and chat template are
  not requantized or numerically modified by this release.

## Paiton overlay

Paiton adds:

- a `gfx1201` shared library containing the compiled graph and RDNA4 kernels;
- a versioned artifact manifest that fixes the target, binary ABI, exact
  parameter shapes/layouts, Quark transformation, and runtime limits;
- a generated `config.json` that registers `PaitonQwen38ForCausalLM` first;
- a text-only contract for TP1, batch 1, and 8,192 tokens;
- installer, verifier, benchmark evidence, and release metadata.

The generated config preserves AMD's original quantization metadata under
`paiton_source_quantization_config`, while setting the generic
`quantization_config` field to `null`. This is intentional: Paiton's strict
loader owns the packed Qronos transformation and prevents the surrounding
runtime from interpreting the checkpoint as another quantization scheme.

The Paiton contract does not expose the source checkpoint's vision/video tower
or MTP path. The second original architecture entry remains for provenance,
but the first and selected architecture is the text-only Paiton class.

## Build provenance

- Artifact compiler revision: `19e5e8bbb82f2cbc6635a0cc7a4d98d0fc7df2d9`
- Artifact runtime/plugin baseline revision:
  `a4954a300e2ea6f7411460090e7afef41454aa24`
- vLLM revision: `39bd959b582c85e78e7e0326d49042ce7c3c07ed`
- Composable Kernel revision:
  `3df477638d05169dbe73f4d58e3a13ad3ca7b5da`
- Artifact SHA256:
  `b3b24c9341c2d28b849e06f5842cfb607af82bb208dea49ef72bffabbe933761`
- Artifact build timestamp: not recorded. File modification time is not used as
  provenance.
- Release source ref:
  `refs/tags/paiton-qwen38-qronos-w4a16-gfx1201-v1`
- Release source revision: injected into the generated release manifest only
  after the clean source commit is tagged; the tracked candidate cannot contain
  its own Git hash.

The artifact revisions describe the bytes that produced the `.so`; the release
source tag describes the later public packaging, cache reuse, server, and
verification code. The private compiler means the binary itself cannot be
reproduced from public source. Reproducible serving inputs are not the same
claim as a reproducible compiler build.
