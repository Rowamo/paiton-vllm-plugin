# Paiton vLLM Plugin

A vLLM platform plugin for running Paiton-compiled models on AMD GPUs.

## Overview

This plugin integrates Paiton-compiled models with vLLM's serving infrastructure, providing:

- **Platform Plugin**: Extends ROCm platform with Paiton-specific optimizations
- **Model Registration**: Registers Paiton model architectures (Llama, Qwen2, etc.)
- **Automatic Weight Mapping**: Handles tensor parallel distribution and FP8 quantization
- **Vendored Runtime Loader**: Includes the runtime Python bindings needed to load compiled `.so` artifacts directly from this repo

## Installation

```bash
# Install the standalone runtime plugin
cd /app/paiton-vllm-plugin
pip install -e .
```

No separate `paiton` runtime repo is required for vLLM serving. The compiler can
stay in `paiton-compiler`, but the runtime-side Python bindings needed to load a
compiled model are now vendored into this plugin repo.

## Usage

### 1. Prepare Your Model Directory

Use a model directory that has already been prepared by the compiler. The runtime plugin expects:

- a PAITON-compatible `config.json`
- one or more compiled `.so` artifacts in the same model directory

### 2. Update Model Configuration

The runtime reads model metadata from the model directory's `config.json`. At minimum it should contain the PAITON architecture entry and, when applicable, `decode_partition_size`:

```json
{
  "architectures": ["PaitonLlamaForCausalLM", "..."],
  "model_type": "llama",
  "decode_partition_size": 256,
  ...
}
```

Compiler-side build and packaging instructions live in [paiton-compiler/README.md](/app/paiton-compiler/README.md).

### 3. Run with vLLM

```bash
# Start the vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /app/paiton-compiler/tmp/Llama-3.1-8B-Instruct-FP8-KV \
    --trust-remote-code

# Or use the Paiton platform explicitly
VLLM_USE_PAITON_PLATFORM=1 python -m vllm.entrypoints.openai.api_server \
    --model /app/paiton-compiler/tmp/Llama-3.1-8B-Instruct-FP8-KV
```

### 4. Run the Offline Benchmark

```bash
cd /app/paiton-vllm-plugin
python3 -m paiton_vllm_plugin.benchmarks.offline_benchmark \
    --model amd/Llama-3.1-8B-Instruct-FP8-KV
```

If the compiled `.so` lives outside `/app/paiton-compiler/tmp/<model-name>`, pass
`--compiled-model-dir /path/to/compiled/model_dir`.

## Artifact Selection

At runtime the plugin resolves artifacts by:

- tensor-parallel size
- `max_num_batched_tokens`
- `hf_config.decode_partition_size`

Supported artifact patterns include:

- legacy: `<model>_tp1.so`
- token-capped: `<model>_tp1_mt16384.so`
- token-capped plus decode partition size: `<model>_tp1_mt16384_ps256.so`

When both `ps256` and `ps512` artifacts exist for the same model and token cap, the plugin expects `decode_partition_size` to be present in `config.json` so it can choose the matching one.

## Supported Models

| Architecture | Paiton Class |
|-------------|--------------|
| Llama, Llama 2, Llama 3 | `PaitonLlamaForCausalLM` |
| Qwen2 | `PaitonQwen2ForCausalLM` |
| Qwen3 | `PaitonQwen3ForCausalLM` |
| Qwen3 MoE | `PaitonQwen3MoeForCausalLM` |

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_USE_PAITON_PLATFORM` | Force use of Paiton platform | `0` |
| `VLLM_DISABLE_PAITON_PLATFORM` | Disable the Paiton platform plugin and use vanilla vLLM platform detection | `0` |

### Platform Detection

The Paiton platform is automatically activated when:
- Running on AMD MI300 series GPUs (gfx942, gfx950)
- `VLLM_USE_PAITON_PLATFORM=1` is set

To explicitly disable the Paiton platform plugin, set:

```bash
export VLLM_DISABLE_PAITON_PLATFORM=1
```

## Plugin Architecture

```
paiton-vllm-plugin/
├── setup.py                    # Entry points registration
├── paiton_vllm_plugin/
│   ├── __init__.py            # Plugin entry points
│   ├── paiton_platform.py     # Platform implementation
│   ├── runtime/               # Vendored Paiton runtime loader bindings
│   └── models/
│       ├── __init__.py
│       ├── paiton_base.py     # Base model class
│       ├── paiton_llama.py    # Llama implementation
│       ├── paiton_qwen.py     # Qwen2 implementation
│       ├── paiton_qwen3.py    # Qwen3 implementation
│       └── paiton_qwen3_moe.py # Qwen3 MoE implementation
```

### Entry Points

The plugin registers two entry points:

1. **Platform Plugin** (`vllm.platform_plugins`): Registers `PaitonPlatform` for ROCm-based execution with Paiton optimizations.

2. **General Plugin** (`vllm.general_plugins`): Registers Paiton model architectures with vLLM's `ModelRegistry`.

## How It Works

### Weight Loading

1. vLLM loads weights from the HuggingFace model
2. `map_pt_params()` transforms weights for Paiton:
   - Fuses QKV and gate/up projections
   - Distributes weights across tensor parallel ranks
   - Converts FP8 weights from `fn` to `fnuz` format for AMD GPUs
3. Weights are set as constants in the Paiton runtime

### Forward Pass

1. vLLM prepares input tensors and attention metadata
2. `forward()` extracts KV cache pointers from vLLM's attention context
3. Paiton runtime executes the compiled model graph from the `.so` selected for the requested token cap and decode partition size
4. Logits are returned for sampling

## Development

### Adding New Models

1. Create a new file in `models/` (e.g., `paiton_mistral.py`)
2. Extend `PaitonModelBase` with model-specific configurations
3. Register the architecture in `__init__.py`

```python
# models/paiton_mistral.py
from paiton_vllm_plugin.models.paiton_base import PaitonModelBase

class PaitonMistralForCausalLM(PaitonModelBase):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
```

### Running Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/
```

## License

Apache-2.0
