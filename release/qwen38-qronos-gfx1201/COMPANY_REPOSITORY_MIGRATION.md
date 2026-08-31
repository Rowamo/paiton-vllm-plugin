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

## One-time GitHub preparation

An Eliovp-BV organization owner creates an empty **public** repository named
`paiton-vllm-plugin` with no generated README/license/gitignore. After the
license and history audit pass:

```bash
cd /path/to/paiton-vllm-plugin

git remote -v
git remote add company git@github.com:Eliovp-BV/paiton-vllm-plugin.git
git remote get-url company

git push company rdna4-qwen38-release:main
git push company <approved-release-tag>
```

If `company` already exists, verify its exact URL rather than rewriting it.
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

## Current external-state gate

As of the release-candidate audit:

- the `Eliovp-BV` GitHub organization exists, but
  `Eliovp-BV/paiton-vllm-plugin` does not yet exist;
- SSH authentication on this host identifies as GitHub user `Eliovp`;
- the `EliovpAI` Hugging Face organization exists;
- this host is not authenticated to Hugging Face.

Repository creation/upload therefore requires an organization owner to create
the empty GitHub repository and a Hugging Face publisher to run `hf auth
login`. These are publication credentials, not code changes, and should never
be committed or pasted into logs.
