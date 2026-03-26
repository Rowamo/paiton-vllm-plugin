# Paiton vLLM Runtime

Run Paiton-prepared models with vLLM on AMD GPUs.

You do not need the compiler to use this runtime. We provide:

- a runtime container image
- a Paiton-prepared model, either as a private Hugging Face repo or a local model directory

There are two supported ways to run the model:

1. Docker image. Recommended.
2. Local install on the host.

## What You Need

- An AMD GPU server with the ROCm driver stack required by the runtime image.
- Access to the runtime image we provide.
- Access to the model we provide.
- A Hugging Face token if the model repo is private.

Example model id used below:

```text
eliovpai/Llama-3.1-8B-Instruct-FP8-KV
```

## Option 1. Docker Image

This is the recommended path.

If the image is private, log in first:

```bash
docker login ghcr.io
docker pull ghcr.io/rowamo/paiton-vllm-plugin:runtime
```

If the model repo is private, authenticate with Hugging Face first:

```bash
export HF_TOKEN=hf_...
```

### Download the Model to the Host First

If you want the model files to stay on the server outside the container, download
them to a host directory first and then mount that directory into the runtime
container.

Authenticate with the Hugging Face CLI:

```bash
hf auth login
```

Download the full model repo into a persistent host directory:

```bash
hf download eliovpai/Llama-3.1-8B-Instruct-FP8-KV \
  --local-dir /srv/models/Llama-3.1-8B-Instruct-FP8-KV
```

Then mount that host directory into the container:

```bash
docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -p 8000:8000 \
  -v /srv/models/Llama-3.1-8B-Instruct-FP8-KV:/models/model:ro \
  ghcr.io/rowamo/paiton-vllm-plugin:runtime
```

Recommended: download the model to a persistent directory on the host first,
then mount that directory into the container. This keeps the model available on
the server even after the container exits.

Alternative: serve the model directly from Hugging Face inside the container:

```bash
docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -e HF_TOKEN=$HF_TOKEN \
  -p 8000:8000 \
  ghcr.io/rowamo/paiton-vllm-plugin:runtime \
  eliovpai/Llama-3.1-8B-Instruct-FP8-KV \
  --kv-cache-dtype fp8 \
  --port 8000
```

If you use the direct Hugging Face path, the downloaded files may only live in
the container filesystem. With `docker run --rm`, that cache is typically not
persisted unless you mount the Hugging Face cache to host storage.

The runtime image entrypoint is already `vllm serve`, so you only pass the model
argument and any extra vLLM flags.

If we provided a local prepared model directory, or if you downloaded the model
repo to the host ahead of time, run:

```bash
docker run --rm \
  --device /dev/kfd \
  --device /dev/dri \
  -p 8000:8000 \
  --group-add video \
  -v /path/to/model:/models/model:ro \
  ghcr.io/rowamo/paiton-vllm-plugin:runtime
```

## Option 2. Local Install

Use this only if you prefer to run directly on the host instead of using the
container image.

Install the plugin into the tested vLLM environment:

```bash
cd /path/to/paiton-vllm-plugin
pip install -e .
```

If the model repo is private, authenticate first:

```bash
export HF_TOKEN=hf_...
```

Serve the model from Hugging Face:

```bash
export VLLM_USE_PAITON_PLATFORM=1

vllm serve eliovpai/Llama-3.1-8B-Instruct-FP8-KV \
  --kv-cache-dtype fp8 \
  --port 8000
```

If we provided a local prepared model directory instead, serve it directly:

```bash
export VLLM_USE_PAITON_PLATFORM=1

vllm serve /path/to/model \
  --kv-cache-dtype fp8 \
  --port 8000
```

## Smoke Test

Once the server is running, verify that it responds:

```bash
curl http://127.0.0.1:8000/v1/models
```

Then send a small request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "eliovpai/Llama-3.1-8B-Instruct-FP8-KV",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 32
  }'
```

## Notes

- The Docker image is the recommended way to run the runtime.
- You do not need to build, publish, or modify container images.
- You do not need the Paiton compiler.
- If you are using a local model directory, it must already contain:
  - model weights
  - tokenizer files
  - `config.json`
  - the compiled Paiton `.so` artifact

## Troubleshooting

- `401` or `403` when loading the model usually means your `HF_TOKEN` is missing
  or does not have access to the private model repo.
- If you are running locally on the host, set `VLLM_USE_PAITON_PLATFORM=1`.
- If you are using a local model directory, make sure the compiled `.so` is in
  the same directory as `config.json`.
