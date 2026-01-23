# gemma3.c

> **⚠️ Work in Progress**: This project is under active development. Inference output may not be fully accurate yet. Contributions and bug reports are welcome!

Pure C inference implementation for Google's Gemma 3 4B IT model.

## Features

- **Zero dependencies** - Pure C11 implementation, no external libraries required
- **Memory-mapped weights** - Efficient loading via mmap, supports BF16 SafeTensors format
- **Full Gemma 3 architecture** - Grouped Query Attention, hybrid local/global attention, SwiGLU MLP
- **SentencePiece tokenizer** - Native protobuf parsing with 262K vocabulary
- **Streaming output** - Real-time token generation with callback support
- **Interactive chat** - Built-in chat mode with Gemma 3 chat template

## Quick Start

```bash
# Build
make

# Run with a prompt
./gemma3 -m ./gemma-3-4b-it -p "Explain quantum computing in simple terms."

# Interactive chat mode
./gemma3 -m ./gemma-3-4b-it -i

# Custom system prompt
./gemma3 -m ./gemma-3-4b-it -i -s "You are a pirate. Respond in pirate speak."
```

## Building

### Requirements

- C11-compatible compiler (GCC, Clang)
- POSIX system (Linux, macOS)
- ~3-4GB RAM for inference

### Build Options

```bash
make          # Release build with -O3
make debug    # Debug build with symbols
make fast     # Aggressive optimizations (-march=native -ffast-math)
make clean    # Remove build artifacts
```

## Model Download

Download the Gemma 3 4B IT model from HuggingFace:

```bash
# Using huggingface-cli
pip install huggingface_hub
huggingface-cli download google/gemma-3-4b-it --local-dir ./gemma-3-4b-it

# Or using git-lfs
git lfs install
git clone https://huggingface.co/google/gemma-3-4b-it

#Or using the python script (recommended)
python download_model.py (rembember to set your HF_TOKEN or pass directly when running the script by including the --token flag)
```

The model directory should contain:
- `model.safetensors` or `model-00001-of-*.safetensors` (weights)
- `tokenizer.model` (SentencePiece vocabulary)

## Usage

### Command Line Options

```
Options:
  -m, --model <path>      Path to model directory (required)
  -p, --prompt <text>     Input prompt for generation
  -i, --interactive       Interactive chat mode
  -s, --system <text>     System prompt for chat mode
  -n, --max-tokens <n>    Maximum tokens to generate (default: 512)
  -t, --temperature <f>   Sampling temperature (default: 0.7)
  -k, --top-k <n>         Top-k sampling (default: 50, 0=disabled)
  --top-p <f>             Top-p sampling (default: 0.9)
  --seed <n>              Random seed (-1 for random)
  -c, --context <n>       Context size (default: 8192)
  -v, --verbose           Verbose output
  -h, --help              Show help message
```

### Library API

```c
#include "gemma3.h"

// Load model
gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");

// Simple generation
gemma3_gen_params params = gemma3_default_params();
char *response = gemma3_generate(ctx, "Hello, world!", &params, NULL, NULL);
printf("%s\n", response);
free(response);

// Chat with streaming
int stream_cb(int token_id, const char *token_str, void *user_data) {
    printf("%s", token_str);
    return 0;  // Return non-zero to stop
}

gemma3_message msgs[] = {
    {GEMMA3_ROLE_USER, "What is the capital of France?"}
};
char *reply = gemma3_chat(ctx, msgs, 1, &params, stream_cb, NULL);
free(reply);

// Cleanup
gemma3_free(ctx);
```

## Architecture

### Gemma 3 4B IT Specifications

| Parameter | Value |
|-----------|-------|
| Vocabulary | 262,208 tokens |
| Hidden size | 2,560 |
| Intermediate size | 10,240 |
| Layers | 34 |
| Attention heads | 8 |
| KV heads | 4 (GQA) |
| Head dimension | 256 |
| Max context | 128K tokens |
| Attention pattern | 5 local : 1 global |
| Sliding window | 1,024 tokens |

### File Structure

```
gemma3.c/
├── gemma3.h              # Public API header
├── gemma3.c              # Main library implementation
├── gemma3_transformer.c  # Transformer forward pass
├── gemma3_safetensors.c  # SafeTensors parser with mmap
├── gemma3_tokenizer.c    # SentencePiece BPE tokenizer
├── gemma3_kernels.c      # CPU compute kernels
├── gemma3_kernels.h      # Kernel declarations
├── main.c                # CLI interface
├── Makefile              # Build configuration
└── README.md             # This file
```

## Memory Usage

Weights are kept in BF16 format (memory-mapped directly from safetensors files) and converted to F32 on-the-fly during computation, minimizing memory usage.

| Component | Size |
|-----------|------|
| Weights (BF16 mmap'd) | ~8 GB (on disk) |
| KV cache (1K context) | ~70 MB |
| Activations | ~100 MB |
| **Total RAM** | **~3 GB** |

For lower memory usage, consider reducing context size with `-c`:
```bash
./gemma3 -m ./gemma-3-4b-it -c 512 -p "Your prompt"
```

## Performance

Performance varies by hardware. On a modern CPU:
- Prefill: ~2-5 tokens/second
- Generation: ~1-3 tokens/second

For better performance:
1. Build with `make fast` for CPU-specific optimizations
2. Use smaller context windows when possible
3. Consider batch prefilling for multiple prompts

## Technical Details

### Attention Pattern

Gemma 3 uses a hybrid attention pattern (every 6th layer is global):
- **Local layers**: Sliding window attention with 1,024 token window, RoPE theta=10,000
- **Global layers** (layers 5, 11, 17, 23, 29): Full attention, RoPE theta=1,000,000

### Normalization

- RMSNorm with epsilon=1e-6
- QK normalization (per-head RMSNorm on Q and K before RoPE)
- Additional pre/post feedforward layer norms (Gemma 3 specific)

### MLP

SwiGLU-style MLP with GELU(tanh approximation):
```
hidden = GELU(gate_proj(x)) * up_proj(x)
output = down_proj(hidden)
```

## Limitations

- Text-only (no vision encoder support yet)
- CPU inference only (no GPU acceleration)
- Not optimized for throughput (designed for clarity)

## License

This implementation is provided under the MIT License. Note that the Gemma model weights have their own license terms from Google.

## Acknowledgments

Inspired by:
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [llama2.c](https://github.com/karpathy/llama2.c)
- [flux2.c](https://github.com/antirez/flux2.c)

## Contributing

Contributions welcome! Areas for improvement:
- SIMD optimizations (AVX2, NEON)
- OpenBLAS/Accelerate backend
- Metal GPU support for macOS
- Quantization support
