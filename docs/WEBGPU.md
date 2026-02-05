# gemma3.c - WebGPU Acceleration Guide

This document describes the WebGPU acceleration support for Gemma 3 inference.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Building](#building)
5. [Architecture](#architecture)
6. [Shaders](#shaders)
7. [API Reference](#api-reference)
8. [Performance](#performance)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The WebGPU backend provides GPU-accelerated inference for Gemma 3 using WebGPU compute shaders. This enables:

- **Cross-platform GPU acceleration** - Works on Vulkan, Metal, and D3D12
- **Browser compatibility** - Same shaders work in browsers with WebGPU
- **BF16 weight support** - Efficient handling of bfloat16 model weights
- **Async execution** - Command batching for improved throughput

### Supported Operations

| Operation | GPU Accelerated | Notes |
|-----------|-----------------|-------|
| Matrix-Vector (BF16) | Yes | Primary hotspot |
| RMSNorm (BF16 weights) | Yes | With Gemma's (1+w) formula |
| GELU activation | Yes | Tanh approximation |
| Softmax | Yes | Numerically stable |
| RoPE | Yes | Position encoding |
| GQA Attention | Yes | Grouped Query Attention |
| Vector Add/Mul | Yes | Element-wise ops |
| Embedding Lookup | Yes | BF16 to F32 |

---

## Requirements

### Hardware

- GPU with Vulkan 1.1+, Metal 2+, or D3D12 support
- Recommended: 8GB+ VRAM for full model

### Software

- **wgpu-native** - Cross-platform WebGPU implementation
  - Download: https://github.com/gfx-rs/wgpu-native/releases
  - Or build from source: https://github.com/gfx-rs/wgpu-native

### Supported Platforms

| Platform | Backend | Status |
|----------|---------|--------|
| Linux | Vulkan | Supported |
| macOS | Metal | Supported |
| Windows | D3D12/Vulkan | Supported |
| Web (WASM) | WebGPU | Planned |

---

## Installation

### Installing wgpu-native

#### Linux (Ubuntu/Debian)

```bash
# Download latest release
wget https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-linux-x86_64-release.zip
unzip wgpu-linux-x86_64-release.zip

# Install to system
sudo cp lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
sudo ldconfig
```

#### macOS

```bash
# Using Homebrew (if available)
# brew install wgpu-native

# Or download manually
wget https://github.com/gfx-rs/wgpu-native/releases/latest/download/wgpu-macos-x86_64-release.zip
unzip wgpu-macos-x86_64-release.zip

# Install
sudo cp lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
```

#### Windows

```powershell
# Download and extract to C:\wgpu
# Add C:\wgpu\lib to PATH
# Set WGPU_DIR=C:\wgpu for building
```

#### Building from Source

```bash
git clone https://github.com/gfx-rs/wgpu-native.git
cd wgpu-native

# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build
cargo build --release

# Install headers and library
sudo cp target/release/libwgpu_native.so /usr/local/lib/
sudo cp ffi/webgpu-headers/webgpu.h /usr/local/include/webgpu/
sudo cp ffi/wgpu.h /usr/local/include/webgpu/
```

---

## Building

### Basic WebGPU Build

```bash
# Build with WebGPU support
make webgpu

# With custom wgpu installation path
make webgpu WGPU_DIR=/path/to/wgpu
```

### WebGPU + Threads

```bash
# Combined GPU and CPU threading
make webgpu-threads
```

### Generate Embedded Shaders

The WGSL shaders can be embedded directly in the binary:

```bash
make shaders
make webgpu
```

### Build Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WGPU_DIR` | `/usr/local` | wgpu-native installation path |
| `CC` | `gcc` | C compiler |

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     gemma3_transformer.c                         │
│                                                                  │
│  matvec_bf16_dispatch() ─────────────────────────────────────┐  │
│       │                                                       │  │
│       ├── USE_WEBGPU ──► gemma3_matvec_bf16_gpu()            │  │
│       │                                                       │  │
│       ├── USE_THREADS ──► gemma3_matvec_bf16_mt()            │  │
│       │                                                       │  │
│       └── default ──► gemma3_matvec_bf16()                   │  │
│                                                               │  │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      gemma3_webgpu.c                             │
│                                                                  │
│  gemma3_gpu_context                                              │
│  ├── WGPUDevice, WGPUQueue                                       │
│  ├── Compute Pipelines (matvec, rmsnorm, gelu, ...)             │
│  ├── Bind Group Layouts                                          │
│  └── Pre-allocated Buffers                                       │
│                                                                  │
│  Kernel Functions:                                               │
│  ├── gemma3_matvec_bf16_gpu()                                    │
│  ├── gemma3_rmsnorm_bf16_gpu()                                   │
│  ├── gemma3_gelu_gpu()                                           │
│  ├── gemma3_softmax_gpu()                                        │
│  ├── gemma3_rope_gpu()                                           │
│  ├── gemma3_gqa_gpu()                                            │
│  └── ...                                                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   shaders/gemma3_kernels.wgsl                    │
│                                                                  │
│  WGSL Compute Shaders:                                           │
│  ├── matvec_bf16_kernel - Matrix-vector with BF16               │
│  ├── rmsnorm_bf16_kernel - RMSNorm with (1+w) formula           │
│  ├── gelu_kernel - GELU tanh approximation                      │
│  ├── softmax_kernel - Numerically stable softmax                │
│  ├── rope_kernel - Rotary position embeddings                   │
│  ├── gqa_kernel - Grouped Query Attention                       │
│  └── ...                                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Model

```
1. Initialization:
   gemma3_gpu_init()
       ├── Create WGPUInstance
       ├── Request WGPUAdapter (high-performance)
       ├── Create WGPUDevice
       ├── Load WGSL shaders
       └── Create compute pipelines

2. Buffer Setup:
   gemma3_gpu_init_buffers()
       └── Pre-allocate GPU buffers for activations

3. Per-Token Inference:
   For each kernel invocation:
       ├── Upload inputs (weights, activations)
       ├── Create bind group
       ├── Encode compute pass
       ├── Submit to queue
       └── (Optionally sync or batch)

4. Cleanup:
   gemma3_gpu_free()
       └── Release all GPU resources
```

---

## Shaders

### Shader Location

- **Source**: `shaders/gemma3_kernels.wgsl`
- **Embedded**: `shaders/gemma3_kernels.wgsl.inc` (generated)

### Key Shader Details

#### BF16 Handling

```wgsl
// Convert BF16 (stored as u32) to F32
fn bf16_to_f32(bf16: u32) -> f32 {
    let bits = bf16 << 16u;
    return bitcast<f32>(bits);
}
```

Weights are stored as packed uint32 (2 BF16 values per element):
```wgsl
let packed = A[(row_offset + k) / 2u];
let a0 = bf16_to_f32(packed & 0xffffu);
let a1 = bf16_to_f32(packed >> 16u);
```

#### Workgroup Sizes

| Kernel | Workgroup Size | Notes |
|--------|----------------|-------|
| matvec_bf16 | 256 | One thread per output row |
| rmsnorm | 256 | Single workgroup reduction |
| gelu | 256 | Parallel element-wise |
| softmax | 256 | Reduction-based |
| rope | 256 | One thread per dimension pair |
| gqa | 256 | One workgroup per head |

#### Shared Memory Usage

```wgsl
var<workgroup> shared: array<f32, 256>;  // For reductions
```

Used for:
- RMSNorm: Sum of squares reduction
- Softmax: Max and sum reductions
- GQA: Attention score processing

---

## API Reference

### Initialization

```c
// Initialize WebGPU context
gemma3_gpu_context *gemma3_gpu_init(void);

// Check if WebGPU is available
int gemma3_gpu_available(void);

// Initialize buffers for model configuration
int gemma3_gpu_init_buffers(gemma3_gpu_context *ctx,
                            int hidden_size,
                            int intermediate_size,
                            int vocab_size,
                            int num_heads,
                            int num_kv_heads,
                            int head_dim,
                            int max_context);

// Cleanup
void gemma3_gpu_free(gemma3_gpu_context *ctx);
```

### Compute Kernels

```c
// Matrix-vector multiplication (BF16 weights)
void gemma3_matvec_bf16_gpu(gemma3_gpu_context *ctx,
                            gemma3_gpu_buffer *y,
                            const uint16_t *A,
                            gemma3_gpu_buffer *x,
                            int M, int K);

// RMSNorm with BF16 weights
void gemma3_rmsnorm_bf16_gpu(gemma3_gpu_context *ctx,
                             gemma3_gpu_buffer *y,
                             gemma3_gpu_buffer *x,
                             const uint16_t *weight,
                             int n, float eps);

// GELU activation (in-place)
void gemma3_gelu_gpu(gemma3_gpu_context *ctx,
                     gemma3_gpu_buffer *x,
                     int n);

// Softmax (in-place)
void gemma3_softmax_gpu(gemma3_gpu_context *ctx,
                        gemma3_gpu_buffer *x,
                        int n);

// RoPE application
void gemma3_rope_gpu(gemma3_gpu_context *ctx,
                     gemma3_gpu_buffer *x,
                     int num_heads,
                     int head_dim,
                     int pos,
                     float theta);

// Grouped Query Attention
void gemma3_gqa_gpu(gemma3_gpu_context *ctx,
                    gemma3_gpu_buffer *output,
                    gemma3_gpu_buffer *q,
                    gemma3_gpu_buffer *k_cache,
                    gemma3_gpu_buffer *v_cache,
                    int n_heads, int n_kv_heads,
                    int seq_len, int head_dim,
                    float scale,
                    gemma3_gpu_buffer *mask);
```

### Buffer Management

```c
// Create GPU buffer
gemma3_gpu_buffer gemma3_gpu_create_buffer(gemma3_gpu_context *ctx,
                                            size_t size,
                                            WGPUBufferUsageFlags usage);

// Upload data to GPU
void gemma3_gpu_write_buffer(gemma3_gpu_context *ctx,
                             gemma3_gpu_buffer *buf,
                             const void *data,
                             size_t size);

// Download data from GPU
void gemma3_gpu_read_buffer(gemma3_gpu_context *ctx,
                            gemma3_gpu_buffer *buf,
                            void *data,
                            size_t size);

// Free buffer
void gemma3_gpu_destroy_buffer(gemma3_gpu_buffer *buf);
```

### Synchronization

```c
// Wait for all GPU operations to complete
void gemma3_gpu_sync(gemma3_gpu_context *ctx);

// Submit pending commands without waiting
void gemma3_gpu_submit(gemma3_gpu_context *ctx);
```

---

## Performance

### Expected Speedups

| Configuration | CPU Baseline | WebGPU | Speedup |
|---------------|--------------|--------|---------|
| Intel i7 (AVX2) | 1x | - | - |
| NVIDIA RTX 3080 | - | 5-10x | Expected |
| Apple M2 (Metal) | - | 3-5x | Expected |
| AMD RX 6800 | - | 4-8x | Expected |

*Note: Actual performance depends on specific hardware and batch sizes.*

### Optimization Tips

1. **Batch Weight Uploads**
   - Upload weights to GPU once at initialization
   - Reuse buffers across tokens

2. **Minimize Synchronization**
   - Use `gemma3_gpu_submit()` for batched execution
   - Only call `gemma3_gpu_sync()` when needed

3. **Buffer Reuse**
   - Use pre-allocated buffers from context
   - Avoid creating/destroying buffers per operation

4. **Context Length**
   - Smaller context = less memory = better GPU utilization
   - Use `-c 2048` for testing

### Memory Requirements

| Context | GPU Memory (Buffers) | Notes |
|---------|---------------------|-------|
| 2048 | ~500 MB | Testing |
| 8192 | ~2 GB | Default |
| 32768 | ~8 GB | Large context |

---

## Troubleshooting

### Common Issues

#### "No suitable GPU adapter found"

- Ensure GPU drivers are up to date
- Check Vulkan/Metal/D3D12 support
- On Linux, install `vulkan-tools` and run `vulkaninfo`

#### "Failed to compile shader module"

- Check WGSL syntax in `shaders/gemma3_kernels.wgsl`
- Ensure wgpu-native version is compatible
- Run with `RUST_BACKTRACE=1` for details

#### Slow Performance

- Verify GPU is being used (check `gemma3_gpu_device_name()`)
- Check for integrated vs discrete GPU selection
- Monitor GPU utilization with system tools

#### Out of Memory

- Reduce context size with `-c` option
- Close other GPU applications
- Check available VRAM

### Debug Build

```bash
# Build with debug info
make debug
export RUST_BACKTRACE=1
./gemma3 -m ./model -p "Test"
```

### Validation Layers

For debugging WebGPU issues:

```bash
# Linux with Vulkan
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
./gemma3 -m ./model -p "Test"
```

---

## Future Work

- [ ] Persistent weight buffers (avoid re-upload)
- [ ] KV cache on GPU
- [ ] Batched inference
- [ ] FP16 accumulation option
- [ ] WebAssembly/Browser support
- [ ] Multi-GPU support

---

## Related Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Main project documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture deep dive
- [BUILD_GUIDE.md](BUILD_GUIDE.md) - Build system details
- [INTERNALS.md](INTERNALS.md) - Implementation deep dive
