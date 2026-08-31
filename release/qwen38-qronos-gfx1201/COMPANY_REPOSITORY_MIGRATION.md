# Company repository migration

The current development repository is owned by Rowamo. The public release must
be owned by the company without disrupting that developer workflow.

## Chosen model: curated company copy

Use a new public repository at
[`Eliovp-BV/paiton-vllm-plugin`](https://github.com/Eliovp-BV/paiton-vllm-plugin).
Keep Rowamo's repository configured as `origin`; add the company repository as
`company`. Do not transfer the existing repository and do not push every local
ref with `--mirror`.

This avoids exposing private/unreviewed branches, pull-request refs, generated
artifacts, or historical secrets. The public repository should contain the
reviewed plugin/runtime source, release documentation, license, notices, tests,
and benchmark harness. The compiler can remain private: consumers of the
precompiled `.so` do not need it.

## Company GitHub import

The public `Eliovp-BV/paiton-vllm-plugin` repository now exists with the
company's Apache-2.0 initial commit, and this checkout keeps it as the separate
`company` remote. After the rights, license, and history audits pass:

```bash
cd /path/to/paiton-vllm-plugin

git remote -v
git remote get-url company

git push company rdna4-qwen38-release:main
git push company <approved-release-tag>
```

Never use `git push --mirror` or `git push --all` for this import.

Recommended repository controls:

- require pull requests and passing tests for `main`;
- block force pushes and branch deletion;
- enable secret scanning, dependency alerts, and signed commits/tags;
- restrict release creation to company maintainers;
- publish the release first as a draft, verify assets, then make it immutable;
- attach provenance/attestation and an SBOM to the exact source tag.

## Hugging Face company model

The existing company organization is
[`EliovpAI`](https://huggingface.co/EliovpAI). The target model repository is:

```text
EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4
```

After a publisher authenticates locally:

```bash
hf auth login

hf repo create \
  EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4 \
  --repo-type model

hf upload \
  EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4 \
  /path/to/complete-verified-model \
  . \
  --repo-type model \
  --commit-message 'Publish Paiton RDNA4 Qwen3.8 release v1'
```

Create and use an immutable signed Hub tag/commit in every serve command. The
complete model directory must include AMD's unchanged weights and metadata plus
the exact Paiton config, `.so`, paired artifact manifest, verifier, license,
notices, and model card. Verify the uploaded checkout on the R9700 before
announcing it.

## Current external state

As of the release-candidate audit:

- the public `Eliovp-BV/paiton-vllm-plugin` repository exists and its initial
  Apache-2.0 commit is preserved in this release branch;
- SSH authentication on this host identifies as GitHub user `Eliovp`;
- the `EliovpAI` Hugging Face organization exists and this host has scoped
  model-repository write access;
- the target Hugging Face model repository has not been created or uploaded;
- this host is authenticated to GHCR, but no release image has been pushed.

Publication credentials are external state, not code, and must never be
committed or pasted into logs. The remaining external operations happen only
after the rights gates and final release bytes are approved.
