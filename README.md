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

### 1. Prepare Your Model

First, compile your model using the Paiton compiler to generate a `.so` file:

```bash
# Example: Compile a Llama model for tensor parallelism size 1
python compile_model.py --model /path/to/Llama-3.1-8B-Instruct --tp 1
```

This creates a file like `Llama-3.1-8B-Instruct_tp1.so` in the model directory.

### 2. Update Model Configuration

Edit your model's `config.json` to use the Paiton architecture:

```json
{
  "architectures": ["PaitonLlamaForCausalLM"],
  "model_type": "llama",
  ...
}
```

### 3. Run with vLLM

```bash
# Start the vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Llama-3.1-8B-Instruct \
    --trust-remote-code

# Or use the Paiton platform explicitly
VLLM_USE_PAITON_PLATFORM=1 python -m vllm.entrypoints.openai.api_server \
    --model /path/to/Llama-3.1-8B-Instruct
```

### 4. Run the Offline Benchmark

```bash
cd /app/paiton-vllm-plugin
python3 -m paiton_vllm_plugin.benchmarks.offline_benchmark \
    --model amd/Llama-3.1-8B-Instruct-FP8-KV
```

If the compiled `.so` lives outside `/app/paiton-compiler/tmp/<model-name>`, pass
`--compiled-model-dir /path/to/compiled/model_dir`.

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

### Platform Detection

The Paiton platform is automatically activated when:
- Running on AMD MI300 series GPUs (gfx942, gfx950)
- `VLLM_USE_PAITON_PLATFORM=1` is set

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
3. Paiton runtime executes the compiled model graph
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
