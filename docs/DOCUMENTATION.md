# gemma3.c - Complete Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Requirements](#system-requirements)
3. [Project Structure](#project-structure)
4. [Core Components](#core-components)
5. [Model Architecture](#model-architecture)
6. [Memory Management](#memory-management)
7. [Performance Characteristics](#performance-characteristics)
8. [Usage Examples](#usage-examples)
9. [Troubleshooting](#troubleshooting)

---

## Project Overview

**gemma3.c** is a from-scratch CPU inference engine for Google's Gemma 3 4B IT (Instruction-Tuned) large language model, implemented entirely in pure C (C11 standard). The project provides a complete, zero-dependency solution for running Gemma 3 inference on standard hardware without requiring GPU acceleration.

### Key Features

| Feature | Description |
|---------|-------------|
| **Pure C Implementation** | No external dependencies beyond the C standard library and POSIX APIs |
| **Memory-Mapped Weights** | Efficient BF16 weight loading via `mmap()` for minimal memory footprint |
| **Native Tokenizer** | Built-in SentencePiece BPE tokenizer with 262K vocabulary |
| **Full Architecture Support** | Complete Gemma 3 transformer: GQA, hybrid attention, SwiGLU, RoPE |
| **Streaming Output** | Token-by-token callback mechanism for real-time text generation |
| **Optional Acceleration** | OpenBLAS and multi-threaded inference support |
| **Chat Interface** | Native support for Gemma 3's instruction-following chat template |

### Design Philosophy

1. **Simplicity First**: Clear, readable code prioritized over micro-optimizations
2. **Zero Dependencies**: Only requires standard C library and POSIX for portability
3. **Memory Efficiency**: BF16 storage with on-the-fly conversion minimizes RAM usage
4. **Modularity**: Clean separation between tokenization, transformer, and I/O

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Linux, macOS (Windows via WSL) |
| **CPU** | x86-64 with SSE2 (AVX2 recommended) |
| **RAM** | 4 GB minimum, 8 GB recommended |
| **Disk** | ~8 GB for model weights |
| **Compiler** | GCC 7+ or Clang 6+ with C11 support |

### Build Dependencies

| Dependency | Required | Purpose |
|------------|----------|---------|
| **GCC/Clang** | Yes | C11 compiler |
| **Make** | Yes | Build system |
| **OpenBLAS** | Optional | BLAS acceleration |
| **pthread** | Optional | Multi-threaded inference |

### Platform Notes

- **Linux**: Full native support with `mmap()` and pthreads
- **macOS**: Full native support (uses Accelerate framework compatible APIs)
- **Windows**: Requires WSL or MinGW (no native `mmap()` support)

---

## Project Structure

```
gemma3.c/
├── README.md                    # Quick start guide
├── Makefile                     # Build system
├── download_model.py            # Model download script
│
├── Public API:
├── gemma3.h                     # Main public interface
├── gemma3_kernels.h             # Compute kernel declarations
├── gemma3_threads.h             # Threading API
│
├── Implementation:
├── main.c                       # CLI application
├── gemma3.c                     # Library core
├── gemma3_transformer.c         # Transformer forward pass
├── gemma3_kernels.c             # Compute kernels
├── gemma3_tokenizer.c           # SentencePiece tokenizer
├── gemma3_safetensors.c         # Weight loader
└── gemma3_threads.c             # Thread pool
```

### File Purposes

| File | Lines | Purpose |
|------|-------|---------|
| `gemma3.h` | 367 | Public API: model loading, generation, tokenization |
| `gemma3.c` | 449 | Main library: ties together all components |
| `gemma3_transformer.c` | 930 | Transformer architecture implementation |
| `gemma3_kernels.c` | 859 | Math operations: matmul, attention, activations |
| `gemma3_tokenizer.c` | 687 | SentencePiece BPE tokenizer |
| `gemma3_safetensors.c` | 721 | SafeTensors parser with mmap |
| `gemma3_threads.c` | ~150 | POSIX thread pool |
| `main.c` | ~600 | Command-line interface |

---

## Core Components

### 1. Model Loading (`gemma3_safetensors.c`)

The SafeTensors loader provides efficient weight access through memory mapping:

```
┌─────────────────────────────────────────────────────────┐
│                    SafeTensors File                      │
├───────────────┬─────────────────────────────────────────┤
│ 8-byte header │           JSON Metadata                  │
│   (size)      │  (tensor names, shapes, offsets)        │
├───────────────┴─────────────────────────────────────────┤
│                                                          │
│              Binary Tensor Data (BF16)                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Key Operations:**
- `st_load()`: Opens all `.safetensors` files in model directory
- `st_find_tensor()`: Locates tensor metadata by name
- `st_get_tensor_data()`: Returns mmap'd pointer to tensor data

**Memory Mapping Benefits:**
- No explicit loading - OS pages in data on demand
- Multiple processes can share mapped pages
- Automatic cleanup on process exit

### 2. Tokenization (`gemma3_tokenizer.c`)

Implements SentencePiece BPE (Byte Pair Encoding) tokenizer:

```
Input Text: "Hello, world!"
     │
     ▼
┌─────────────────────────────────────┐
│  UTF-8 Character Segmentation       │
│  + Word Boundary Detection (▁)      │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│  BPE Merge Loop                     │
│  (merge pairs by score until done)  │
└─────────────────────────────────────┘
     │
     ▼
Output Tokens: [2, 17534, 235269, 2134, 235341]
```

**Features:**
- 262,208 vocabulary tokens
- Byte fallback for unknown characters (`<0xNN>` format)
- Special tokens: `<bos>`, `<eos>`, `<start_of_turn>`, `<end_of_turn>`
- Word boundary marker: `▁` (U+2581)

### 3. Transformer (`gemma3_transformer.c`)

Implements the complete Gemma 3 transformer forward pass:

```
Token ID
    │
    ▼
┌─────────────────────────────────────┐
│        Embedding Lookup             │
│   (BF16 → F32, scaled by √hidden)   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│        Layer 0..33 (x34)            │
│  ┌───────────────────────────────┐  │
│  │  RMSNorm (input)              │  │
│  │  Multi-Head Attention (GQA)   │  │
│  │  RMSNorm (post-attention)     │  │
│  │  Residual Connection          │  │
│  │  RMSNorm (pre-feedforward)    │  │
│  │  SwiGLU MLP                   │  │
│  │  RMSNorm (post-feedforward)   │  │
│  │  Residual Connection          │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│         Final RMSNorm               │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│   Output Projection (Tied Embed)    │
│         → [vocab_size] logits       │
└─────────────────────────────────────┘
```

### 4. Compute Kernels (`gemma3_kernels.c`)

Provides optimized implementations for all mathematical operations:

| Operation | Description | Optimization |
|-----------|-------------|--------------|
| `gemma3_matvec_bf16` | Matrix-vector with BF16 weights | AVX2 + BLAS |
| `gemma3_rmsnorm_bf16` | RMS normalization | Gemma's (1+w) formula |
| `gemma3_gqa` | Grouped Query Attention | Pre-allocated buffers |
| `gemma3_rope_apply_precomputed` | Rotary embeddings | Precomputed sin/cos |
| `gemma3_gelu_tanh` | GELU activation | Tanh approximation |
| `gemma3_softmax_inplace` | Numerically stable softmax | Max subtraction |

---

## Model Architecture

### Gemma 3 4B Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vocab_size` | 262,208 | Vocabulary size |
| `hidden_size` | 2,560 | Hidden dimension |
| `intermediate_size` | 10,240 | MLP intermediate dimension (4x hidden) |
| `num_layers` | 34 | Number of transformer layers |
| `num_heads` | 8 | Number of attention heads |
| `num_kv_heads` | 4 | Number of KV heads (GQA: 2:1 ratio) |
| `head_dim` | 256 | Dimension per head |
| `max_context` | 131,072 | Maximum context length (128K) |
| `sliding_window` | 1,024 | Local attention window size |

### Attention Mechanism

Gemma 3 uses a **hybrid attention** pattern combining local and global attention:

```
Layer Index:  0   1   2   3   4   5   6   7   8   9  10  11  ...
Attention:    L   L   L   L   L   G   L   L   L   L   L   G   ...

L = Local (sliding window, θ=10,000)
G = Global (full causal, θ=1,000,000)
```

**Pattern:** Every 6th layer (indices 5, 11, 17, 23, 29) uses global attention.

**Grouped Query Attention (GQA):**
- 8 query heads share 4 KV heads
- Each KV head serves 2 query heads
- Reduces KV cache memory by 50%

### MLP Architecture (SwiGLU)

```
Input x [hidden_size=2560]
    │
    ├─────────────────┬─────────────────┐
    │                 │                 │
    ▼                 ▼                 │
gate_proj(x)     up_proj(x)            │
[intermediate]   [intermediate]         │
    │                 │                 │
    ▼                 │                 │
  GELU               │                 │
    │                 │                 │
    └────────┬────────┘                 │
             │                          │
             ▼                          │
         gate × up                      │
             │                          │
             ▼                          │
        down_proj                       │
             │                          │
             ▼                          │
         + ──────────────────────────←──┘  (residual)
             │
             ▼
Output [hidden_size=2560]
```

### Normalization

Gemma 3 uses **RMSNorm** (Root Mean Square Normalization) with a unique formula:

```
Standard RMSNorm: y = x * rsqrt(mean(x²) + ε) * weight

Gemma RMSNorm:    y = x * rsqrt(mean(x²) + ε) * (1 + weight)
```

The `(1 + weight)` formulation provides better gradient flow during training.

---

## Memory Management

### Weight Storage

| Component | Size Formula | Gemma 3 4B |
|-----------|--------------|------------|
| Embeddings | vocab × hidden × 2 | 1.28 GB |
| Attention (per layer) | (Q+K+V+O) × hidden² × 2 | ~106 MB |
| MLP (per layer) | 3 × inter × hidden × 2 | ~150 MB |
| Layer Norms (per layer) | 6 × hidden × 2 | ~30 KB |
| **Total** | - | **~8 GB** |

### Runtime Buffers

| Buffer | Size | Purpose |
|--------|------|---------|
| Hidden state | hidden × 4 | Current activations |
| Q/K/V | heads × head_dim × 4 | Attention projections |
| MLP intermediate | inter × 4 | SwiGLU computation |
| Logits | vocab × 4 | Output distribution |
| KV Cache | varies | Cached key/value pairs |

### KV Cache Allocation

```c
// Global layers: full context
if (is_global_layer) {
    cache_size = max_context × num_kv_heads × head_dim × sizeof(float);
}
// Local layers: sliding window only
else {
    cache_size = sliding_window × num_kv_heads × head_dim × sizeof(float);
}
```

**Memory Savings:** Local layers use ring buffers, reducing total KV cache by ~83%.

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Gemma 3 (per token) |
|-----------|------------|---------------------|
| Embedding | O(hidden) | 2,560 ops |
| Attention QKV | O(hidden²) | ~20M ops |
| Attention scores | O(seq × head_dim) | variable |
| MLP | O(hidden × inter) | ~52M ops |
| **Total per layer** | - | ~75M ops |
| **All 34 layers** | - | ~2.5B ops |

### Expected Performance (CPU)

| Configuration | Prefill (tok/s) | Decode (tok/s) |
|---------------|-----------------|----------------|
| Default (`make`) | 2-5 | 1-3 |
| Fast (`make fast`) | 3-6 | 2-4 |
| Threaded (`make threads`) | 4-8 | 3-5 |
| BLAS+Threads | 6-12 | 4-8 |

*Performance varies significantly based on CPU model, cache size, and memory bandwidth.*

---

## Usage Examples

### Basic Text Generation

```c
#include "gemma3.h"

int main() {
    // Load model
    gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");
    if (!ctx) {
        fprintf(stderr, "Failed: %s\n", gemma3_get_error());
        return 1;
    }

    // Generate with default parameters
    gemma3_gen_params params = gemma3_default_params();
    char *output = gemma3_generate(ctx, "Explain quantum computing:", &params, NULL, NULL);

    printf("%s\n", output);
    free(output);
    gemma3_free(ctx);
    return 0;
}
```

### Streaming Output

```c
int stream_callback(int token_id, const char *token_str, void *user_data) {
    printf("%s", token_str);
    fflush(stdout);
    return 0;  // Return non-zero to stop generation
}

// In main:
gemma3_generate(ctx, prompt, &params, stream_callback, NULL);
```

### Chat Completion

```c
gemma3_message messages[] = {
    { GEMMA3_ROLE_SYSTEM, "You are a helpful assistant." },
    { GEMMA3_ROLE_USER, "What is the capital of France?" },
};

char *response = gemma3_chat(ctx, messages, 2, &params, stream_callback, NULL);
```

### Custom Parameters

```c
gemma3_gen_params params = {
    .max_tokens = 256,
    .temperature = 0.8f,     // Higher = more creative
    .top_k = 40,             // Consider top 40 tokens
    .top_p = 0.95f,          // Nucleus sampling
    .seed = 42,              // Reproducible output
    .stop_on_eos = 1,
    .greedy = 0,
};
```

### CLI Usage

```bash
# Basic generation
./gemma3 -m ./gemma-3-4b-it -p "Write a haiku about programming"

# With custom parameters
./gemma3 -m ./gemma-3-4b-it \
    -p "Explain machine learning" \
    -n 512 \
    -t 0.7 \
    -k 50 \
    --top-p 0.9 \
    --seed 42

# With system prompt
./gemma3 -m ./gemma-3-4b-it \
    -s "You are a pirate. Respond in pirate speak." \
    -p "Tell me about the weather"

# Reduced context for memory savings
./gemma3 -m ./gemma-3-4b-it -c 512 -p "Hello"
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Failed to load model" | Missing files | Ensure all `.safetensors` and `tokenizer.model` exist |
| "mmap failed" | Insufficient memory | Reduce context with `-c 512` |
| Slow generation | No optimizations | Build with `make blas-threads` |
| Garbage output | Corrupted weights | Re-download model |
| "Context overflow" | Prompt too long | Increase `-c` or shorten prompt |

### Debug Options

```bash
# Verbose mode (shows token IDs)
./gemma3 -m ./gemma-3-4b-it -p "Hello" -v

# Tokenization debug
./gemma3 -m ./gemma-3-4b-it -p "Hello world" --tokenize

# Show top logits
./gemma3 -m ./gemma-3-4b-it -p "Hello" --logits
```

### Memory Issues

If encountering out-of-memory errors:

1. Reduce context size: `-c 1024` or `-c 512`
2. Close other applications
3. Use swap space (slower but works)
4. Consider a machine with more RAM

### Build Issues

```bash
# Missing OpenBLAS
sudo apt install libopenblas-dev  # Ubuntu/Debian
brew install openblas             # macOS

# Compilation errors
make clean && make                # Clean rebuild

# Wrong compiler
CC=gcc make                       # Force GCC
CC=clang make                     # Force Clang
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture explanation
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API documentation
- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Build system details
- [INTERNALS.md](INTERNALS.md) - Implementation deep dive

---

*This documentation covers gemma3.c version 0.1.0*
