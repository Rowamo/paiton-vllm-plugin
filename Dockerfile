ARG VLLM_BASE_IMAGE=rocm/vllm-dev:nightly_main_20260312
FROM ${VLLM_BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/Rowamo/paiton-vllm-plugin" \
      org.opencontainers.image.description="Runtime image for serving Paiton-compiled models with vLLM on ROCm"

WORKDIR /app/paiton-vllm-plugin

COPY . /app/paiton-vllm-plugin

# Keep the tested torch/vLLM stack from the base image intact.
RUN python3 -m pip install --no-cache-dir --no-deps /app/paiton-vllm-plugin

ENV PYTHONUNBUFFERED=1 \
    VLLM_USE_PAITON_PLATFORM=1 \
    VLLM_DISABLE_PAITON_PLATFORM=0

EXPOSE 8000
VOLUME ["/models"]

ENTRYPOINT ["vllm", "serve"]
CMD ["/models/model", "--kv-cache-dtype", "fp8", "--host", "0.0.0.0", "--port", "8000"]
