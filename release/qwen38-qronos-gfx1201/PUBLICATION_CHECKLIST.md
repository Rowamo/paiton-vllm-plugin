# Publication checklist

The technical artifact is ready for packaging, but the following gates are
mandatory before calling it a public release.

Canonical public owners:

- GitHub: `Eliovp-BV/paiton-vllm-plugin`
- Hugging Face: `EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4`
- Rowamo repositories remain development origins and are not transferred.

## 1. Rights and licensing

- [x] Paiton's rights holder selected Apache-2.0 in the company repository's
      initial commit.
- [x] Add the complete root `LICENSE` to `paiton-vllm-plugin` and preserve that
      company commit in the imported history.
- [ ] Confirm the private compiler's generated artifact may be distributed
      under the release license; publishing compiler source is not required.
- [ ] Confirm contributor rights for the compiler, plugin, generated runtime,
      and kernels.
- [ ] Review `THIRD_PARTY_NOTICES.md` against the exact generated source and
      linked binary.
- [ ] Ship complete license texts and preserve all required copyright notices.
- [ ] Confirm that model-card metadata and redistribution of AMD's unchanged
      Apache-2.0 checkpoint meet notice and modification requirements.

## 2. Freeze exact bytes and provenance

- [ ] Tag the exact compiler source that produced artifact SHA256
      `b3b24c9341c2d28b849e06f5842cfb607af82bb208dea49ef72bffabbe933761`.
- [ ] Tag the exact runtime/plugin source used for final serving validation.
- [x] Distinguish `artifact_compiler_revision`,
      `artifact_plugin_revision`, the unrecorded `artifact_built_at`,
      `release_source_ref`, generated `release_source_revision`, and
      `packaged_at` in the release manifest without creating a Git-hash
      self-reference.
- [ ] Record the full source model ID and revision
      `649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2` everywhere.
- [ ] Rebuild the qualified runtime container from the final plugin tag and
      publish it by immutable image digest.
- [ ] Build `Dockerfile.qwen38-rdna4`, verify that its zero-argument entrypoint
      selects the immutable EliovpAI model revision and exact runtime contract,
      and publish both the friendly tag and immutable digest.
- [x] Generate an SPDX document scoped to the compiled artifact, including its
      source/build relationships and exact direct ELF runtime dependencies.
- [ ] Generate a separate dependency-level SBOM from the final OCI image.
- [x] Record an unsigned in-toto provenance statement for the compiled outputs
      without claiming a public or reproducible compiler build.
- [ ] Sign or attest the final archive separately; do not describe the unsigned
      build record as a third-party attestation.

## 3. Prepare the company GitHub repository

- [x] Create the public `Eliovp-BV/paiton-vllm-plugin` repository under the
      company organization with its Apache-2.0 initial commit.
- [ ] Import only the reviewed public plugin tree. Do not `git push --mirror`
      development refs, private branches, pull-request refs, or unreviewed
      history.
- [x] Add the company repository as a separate `company` remote; retain
      Rowamo's repository as `origin` for development.
- [ ] Enable branch protection, immutable releases, dependency/security
      scanning, and release attestations.

See [`COMPANY_REPOSITORY_MIGRATION.md`](COMPANY_REPOSITORY_MIGRATION.md).

## 4. Build the GitHub asset

- [ ] Assemble the bundle from the declared `.so`, paired artifact manifest,
      and generated config.
- [ ] Set ordinary files and the `.so` to mode `0644`; scripts may be `0755`.
- [ ] Generate `SHA256SUMS` for every file and a separate hash for the outer
      archive.
- [ ] Use the name
      `paiton-qwen38-qronos-w4a16-v1-linux-x86_64-rocm7.14-gfx1201-tp1-ctx8192.tar.zst`.
- [ ] Extract the archive into a clean directory and run
      `verify_release.py` before uploading.
- [ ] Create a draft GitHub Release from the exact source tag, attach assets,
      verify them, add provenance/attestation, then make the release immutable.

## 5. Publish the Hugging Face model

- [x] Authenticate a publisher with write access to the existing `EliovpAI`
      Hugging Face organization.
- [ ] Create
      `EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4`.
- [ ] Upload a complete model repository, not an artifact-only repository:
      unchanged weights/tokenizer, generated config, `.so`, artifact manifest,
      verifier, notices, and model card.
- [ ] Make the executable-code warning and exact support matrix prominent.
- [ ] Include `base_model`, `library_name`, `pipeline_tag`, license, and RDNA4
      tags after license review.
- [ ] Sign the publishing commit and create an immutable Hub tag.
- [ ] Pin every command to that commit or tag rather than `main`.
- [ ] Verify that the `.so` and weight hashes exactly match the GitHub release
      and AMD source revision.

## 6. Clean-room serving gate

- [x] Validate the release-candidate image entrypoint on the R9700 with the
      verified complete model directory: model load, `/health`, and chat
      generation all passed. This is recorded in
      `results/one-line-container-smoke.json`.
- [ ] On the R9700, download only the published container and HF/GitHub bytes.
- [ ] Run the documented one-line `docker run` command from an empty Docker
      volume and prove first-run download, health, generation, restart, and
      cache reuse.
- [ ] Verify all hashes before loading the `.so`.
- [ ] Start `vllm serve` with the exact TP1/batch1/context8192 contract.
- [ ] Run the health and chat smoke tests.
- [ ] Run the five-workload controlled benchmark in fresh processes and retain
      provider-selection logs and token hashes.
- [ ] Run the planned 1,024-request ShareGPT serving benchmark at least three
      times per successful provider, preserving failures as failures.
- [ ] Confirm the public-compatible and unmodified AITER commands work from the
      published instructions; do not apply local source patches.
- [ ] Cross-check every table in the blog against the final raw JSON.

## 7. Cross-link and announce

- [ ] Add final GitHub release, HF model/tag, runtime image digest, source tags,
      raw results, and blog URLs to each publication surface.
- [ ] State explicitly that the `.so` is executable code, not weights.
- [ ] State every limitation: R9700/gfx1201 only, ROCm 7.14 stack, TP1, batch1,
      8192 tokens, text-only, no MTP, and no AWQ/GLM/SGLang claim.
- [ ] Label the 16-request online run a smoke test until the larger repeated
      serving run is published.
