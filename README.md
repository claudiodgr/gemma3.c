# gemma3.c

`gemma3.c` is a **from-scratch CPU inference engine** for the *Gemma 3 4B IT* model.

## Highlights

* **100% Pure C (C11)** - zero external dependencies
* **Full Gemma 3 architecture** - GQA, hybrid attention, SwiGLU
* **Memory-mapped weights** - BF16 SafeTensors
* **Native SentencePiece tokenizer** - 262K vocab
* **Streaming output** - token-by-token callbacks
* **Interactive chat mode**
* **CLI + Library API**
* **Cross-platform** - Linux, Windows (native), macOS
* **OpenBLAS support** (optional) - BLAS-accelerated matrix operations
* **Multi-threaded inference** - Thread pool for parallel computation
* **WebGPU GPU acceleration** (optional) - GPU compute via wgpu-native

---

## Quick Start

### 1. Download model

```bash
export HF_TOKEN=your_token_here
pip install huggingface_hub
python download_model.py
```

### 2. Build

**CMake (Recommended - All Platforms):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**Makefile (Linux/macOS):**

```bash
make
```

### 3. Run

```bash
# Single prompt
./build/gemma3 -m ./gemma-3-4b-it -p "Explain quantum computing simply."

# Interactive chat
./build/gemma3 -m ./gemma-3-4b-it -i
```

---

## Build Options

### CMake (Cross-Platform)

```bash
# Basic release build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# With threading
cmake -B build -DGEMMA3_USE_THREADS=ON
cmake --build build

# With OpenBLAS + threading (best CPU performance)
cmake -B build -DGEMMA3_USE_BLAS=ON -DGEMMA3_USE_THREADS=ON -DGEMMA3_NATIVE=ON
cmake --build build

# With WebGPU GPU acceleration
cmake -B build -DGEMMA3_USE_WEBGPU=ON
cmake --build build
```

**CMake Presets:**

```bash
cmake --preset release        # Basic release
cmake --preset threads        # Multi-threaded
cmake --preset blas-threads   # BLAS + threads
cmake --preset webgpu         # GPU acceleration
cmake --preset full           # All features
```

### Makefile (Linux/macOS)

```bash
make              # Release build (default)
make debug        # Debug symbols
make fast         # Native optimizations (-march=native -ffast-math)
make threads      # Thread pool parallelization
make blas         # OpenBLAS acceleration
make blas-threads # OpenBLAS + threads (best performance)
make webgpu       # WebGPU GPU acceleration
make clean        # Remove build artifacts
```

---

## Platform Support

| Platform | CMake | Makefile | Notes |
|----------|-------|----------|-------|
| Linux (x86_64) | Yes | Yes | Full support |
| Linux (ARM64) | Yes | Yes | Full support |
| macOS (Intel) | Yes | Yes | Full support |
| macOS (Apple Silicon) | Yes | Yes | Full support |
| Windows (MSVC) | Yes | No | Visual Studio 2019+ |
| Windows (MinGW) | Yes | No | MSYS2/MinGW-w64 |
| Windows (WSL) | Yes | Yes | Uses Linux build |

---

## Dependencies

### Required

* C11 compiler (GCC 7+, Clang 6+, MSVC 2019+)
* CMake 3.16+ (for CMake builds)

### Optional

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| OpenBLAS | BLAS acceleration | `apt install libopenblas-dev` / `brew install openblas` |
| wgpu-native | GPU acceleration | See [WebGPU Setup](#webgpu-setup) |

---

## WebGPU Setup

For GPU acceleration, download wgpu-native libraries:

```bash
# Linux/macOS
./scripts/download_webgpu.sh

# Windows (PowerShell)
.\scripts\download_webgpu.ps1
```

Then build with WebGPU enabled:

```bash
cmake -B build -DGEMMA3_USE_WEBGPU=ON
cmake --build build
```

---

## CLI Options

```
-m <path>    Model directory
-p <text>    Prompt
-i           Interactive mode
-s <text>    System prompt
-n <n>       Max tokens
-t <f>       Temperature
-k <n>       Top-k
--top-p <f>  Top-p
-c <n>       Context size
--seed <n>   RNG seed
-v           Verbose
```

---

## Library Example

```c
gemma3_ctx *ctx = gemma3_load_dir("./gemma-3-4b-it");

gemma3_gen_params params = gemma3_default_params();
char *out = gemma3_generate(ctx, "Hello!", &params, NULL, NULL);
printf("%s\n", out);
free(out);

gemma3_free(ctx);
```

---

## Model Specs

| Param   | Value              |
| ------- | ------------------ |
| Vocab   | 262,208            |
| Layers  | 34                 |
| Hidden  | 2,560              |
| Heads   | 8 (4 KV, GQA)      |
| Context | 128K               |
| Pattern | 5 local : 1 global |

---

## Memory

* Weights: ~8 GB on disk (BF16)
* Runtime RAM: **~3 GB total**

Reduce usage with smaller context:

```bash
./build/gemma3 -m ./gemma-3-4b-it -c 512 -p "Hello"
```

---

## Performance

### CPU

* Prefill: ~2-5 tok/s
* Generation: ~1-3 tok/s

For better performance:

```bash
cmake -B build -DGEMMA3_NATIVE=ON -DGEMMA3_USE_THREADS=ON -DGEMMA3_USE_BLAS=ON
cmake --build build
```

### GPU (WebGPU)

With GPU acceleration enabled, expect significantly faster inference depending on your GPU.

---

## Documentation

* [Build Guide](docs/BUILD_GUIDE.md) - Detailed build instructions
* [Architecture](docs/ARCHITECTURE.md) - Code structure and design
* [API Reference](docs/API_REFERENCE.md) - Library API documentation
* [WebGPU Guide](docs/WEBGPU.md) - GPU acceleration setup

---

## Limitations

* Text only (no multimodal)
* No quantization (yet)
* WebGPU requires compatible GPU drivers

---

## License

MIT License.
Model weights under Google's Gemma license.

---

*If you ever wanted to see Gemma 3 breathe in pure C, this is it.*
