# gemma3.c - Build Guide

Complete guide for building, configuring, and installing gemma3.c.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build Targets](#build-targets)
4. [Build Configuration](#build-configuration)
5. [Platform-Specific Instructions](#platform-specific-instructions)
6. [Model Download](#model-download)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| GCC or Clang | 7+ / 6+ | C11 compiler |
| GNU Make | 3.8+ | Build system |
| Python 3 | 3.7+ | Model download script |

### Optional Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| OpenBLAS | BLAS acceleration | `apt install libopenblas-dev` |
| pthreads | Multi-threading | Usually included with libc |
| huggingface_hub | Model download | `pip install huggingface_hub` |

### Disk Space Requirements

| Component | Size |
|-----------|------|
| Source code | ~1 MB |
| Build artifacts | ~5 MB |
| Model weights | ~8 GB |
| **Total** | **~8 GB** |

### Memory Requirements

| Configuration | RAM Required |
|---------------|--------------|
| Minimum (context=512) | ~3 GB |
| Default (context=8192) | ~4 GB |
| Maximum (context=131072) | ~16 GB |

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/your-repo/gemma3.c.git
cd gemma3.c

# 2. Build
make

# 3. Download model (requires HuggingFace token)
export HF_TOKEN=your_token_here
pip install huggingface_hub
python download_model.py

# 4. Run
./gemma3 -m ./gemma-3-4b-it -p "Hello, world!"
```

---

## Build Targets

### Available Targets

| Target | Command | Description |
|--------|---------|-------------|
| Release | `make` | Default optimized build |
| Debug | `make debug` | Debug symbols, no optimization |
| Fast | `make fast` | Native CPU optimizations |
| Threads | `make threads` | Multi-threaded inference |
| BLAS | `make blas` | OpenBLAS acceleration |
| BLAS+Threads | `make blas-threads` | Best performance |
| Clean | `make clean` | Remove build artifacts |
| Help | `make help` | Show available targets |

### Target Details

#### Release Build (Default)

```bash
make
```

**Flags:** `-O3 -DNDEBUG`

Standard optimized build suitable for most use cases.

#### Debug Build

```bash
make debug
```

**Flags:** `-g -O0 -DDEBUG`

Includes debug symbols for gdb/lldb. No optimization for accurate debugging.

#### Fast Build

```bash
make fast
```

**Flags:** `-O3 -march=native -ffast-math -DNDEBUG`

Uses CPU-specific optimizations. Binary may not work on other machines.

#### Threaded Build

```bash
make threads
```

**Flags:** `-DUSE_THREADS`
**Libs:** `-lpthread`
**Additional source:** `gemma3_threads.c`

Enables multi-core parallelization of matrix operations.

#### BLAS Build

```bash
make blas
```

**Flags:** `-DUSE_BLAS`
**Libs:** `-lopenblas`

Uses OpenBLAS for optimized BLAS routines (sgemm, sgemv, sdot).

#### BLAS + Threads Build

```bash
make blas-threads
```

**Flags:** `-DUSE_BLAS -DUSE_THREADS`
**Libs:** `-lopenblas -lpthread`

Combines OpenBLAS with thread pool for maximum performance.

---

## Build Configuration

### Compiler Selection

```bash
# Use GCC (default)
make CC=gcc

# Use Clang
make CC=clang
```

### Custom Flags

The Makefile uses these flag categories:

```makefile
CFLAGS_BASE    = -Wall -Wextra -Wpedantic -std=c11 -MMD -MP
CFLAGS_RELEASE = -O3 -DNDEBUG
CFLAGS_DEBUG   = -g -O0 -DDEBUG
CFLAGS_FAST    = -O3 -march=native -ffast-math -DNDEBUG

LDFLAGS_BASE   = -lm
```

### Build Directory Structure

```
build/
├── release/          # make
│   ├── gemma3.o
│   ├── main.o
│   └── ...
├── debug/            # make debug
│   └── ...
├── fast/             # make fast
│   └── ...
├── threads/          # make threads
│   └── ...
├── blas/             # make blas
│   └── ...
└── blas-threads/     # make blas-threads
    └── ...
```

### Preprocessor Defines

| Define | Effect |
|--------|--------|
| `NDEBUG` | Disables assert() |
| `DEBUG` | Enables debug logging |
| `USE_BLAS` | Enables OpenBLAS routines |
| `USE_THREADS` | Enables thread pool |

### Compiler Auto-Detection

The code automatically detects certain CPU features:

```c
#ifdef __AVX2__
// AVX2 SIMD optimizations enabled
#endif
```

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install build tools
sudo apt update
sudo apt install build-essential

# Install optional dependencies
sudo apt install libopenblas-dev  # For BLAS support

# Build
make blas-threads
```

### Linux (Fedora/RHEL)

```bash
# Install build tools
sudo dnf install gcc make

# Install optional dependencies
sudo dnf install openblas-devel

# Build
make blas-threads
```

### Linux (Arch)

```bash
# Install build tools
sudo pacman -S base-devel

# Install optional dependencies
sudo pacman -S openblas

# Build
make blas-threads
```

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install optional dependencies
brew install openblas

# Build (note: may need to specify OpenBLAS path)
LDFLAGS="-L/opt/homebrew/opt/openblas/lib" \
CFLAGS="-I/opt/homebrew/opt/openblas/include" \
make blas-threads
```

### Windows (WSL)

Windows Subsystem for Linux is the recommended approach:

```bash
# Install WSL (in PowerShell as Administrator)
wsl --install

# Inside Ubuntu WSL:
sudo apt update
sudo apt install build-essential libopenblas-dev

# Build
make blas-threads
```

### Windows (MinGW)

MinGW builds work but have limitations (no mmap):

```bash
# Install MSYS2 from https://www.msys2.org/

# In MSYS2 terminal:
pacman -S mingw-w64-x86_64-gcc make

# Build (limited functionality)
make
```

**Note:** MinGW builds don't support memory mapping. The SafeTensors loader will use a fallback that loads entire files into memory.

---

## Model Download

### Using the Download Script

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Install required Python package
pip install huggingface_hub

# Download model
python download_model.py
```

### Script Options

```bash
python download_model.py --help

Options:
  --token TOKEN      HuggingFace API token
  --repo REPO        Model repository (default: google/gemma-3-4b-it)
  --output DIR       Output directory (default: ./gemma-3-4b-it)
```

### Manual Download

Alternatively, use the HuggingFace CLI:

```bash
# Install CLI
pip install huggingface-cli

# Login
huggingface-cli login

# Download
huggingface-cli download google/gemma-3-4b-it --local-dir ./gemma-3-4b-it
```

### Required Model Files

After download, verify these files exist:

```
gemma-3-4b-it/
├── model-00001-of-00002.safetensors  # ~4 GB
├── model-00002-of-00002.safetensors  # ~4 GB
├── tokenizer.model                    # ~4 MB
├── config.json                        # Model config
└── tokenizer_config.json              # Tokenizer config
```

---

## Verification

### Build Verification

```bash
# Check binary exists
ls -la gemma3

# Check version
./gemma3 --help
```

### Quick Test

```bash
# Simple generation test
./gemma3 -m ./gemma-3-4b-it -p "2+2=" -n 10
# Expected: "4" or similar

# Verbose mode
./gemma3 -m ./gemma-3-4b-it -p "Hello" -n 5 -v
```

### Tokenization Test

```bash
# Test tokenizer
./gemma3 -m ./gemma-3-4b-it -p "Hello, world!" --tokenize
# Shows token IDs
```

### Performance Test

```bash
# Time a generation
time ./gemma3 -m ./gemma-3-4b-it -p "Explain quantum computing" -n 100

# Compare build modes
make clean && make && time ./gemma3 -m ./gemma-3-4b-it -p "Test" -n 50
make clean && make blas-threads && time ./gemma3 -m ./gemma-3-4b-it -p "Test" -n 50
```

---

## Troubleshooting

### Compilation Errors

#### "stdio.h not found"

Install development headers:
```bash
# Ubuntu/Debian
sudo apt install libc6-dev

# Fedora
sudo dnf install glibc-devel
```

#### "cblas.h not found"

Install OpenBLAS development files:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel

# macOS
brew install openblas
```

#### "pthread.h not found"

pthreads should be included with libc. Try:
```bash
make threads LDFLAGS="-lpthread"
```

### Linker Errors

#### "undefined reference to cblas_sgemv"

OpenBLAS not linked properly:
```bash
make blas LDFLAGS="-L/path/to/openblas/lib -lopenblas"
```

#### "undefined reference to sqrt"

Math library not linked:
```bash
make LDFLAGS="-lm"
```

### Runtime Errors

#### "Failed to load model"

Check model directory structure:
```bash
ls -la ./gemma-3-4b-it/
# Should show .safetensors files and tokenizer.model
```

#### "mmap failed"

Not enough address space or permissions:
```bash
# Check available memory
free -h

# Try with reduced context
./gemma3 -m ./gemma-3-4b-it -c 512 -p "Hello"
```

#### "Segmentation fault"

Often memory-related:
```bash
# Run with debug build
make debug
gdb ./gemma3
(gdb) run -m ./gemma-3-4b-it -p "Hello"
(gdb) bt  # Show backtrace
```

### Performance Issues

#### Slow generation

Try optimized builds:
```bash
# Best performance
make blas-threads

# Or native optimizations
make fast
```

#### High memory usage

Reduce context size:
```bash
./gemma3 -m ./gemma-3-4b-it -c 1024 -p "Hello"
```

### Build System Issues

#### "Nothing to be done"

Force rebuild:
```bash
make clean
make
```

#### Wrong build mode

Check the binary was built with expected flags:
```bash
# Rebuild from scratch
rm -rf build gemma3
make blas-threads
```

---

## Advanced Configuration

### Custom OpenBLAS Location

```bash
# macOS with Homebrew
export OPENBLAS_DIR=/opt/homebrew/opt/openblas
make blas CFLAGS="-I$OPENBLAS_DIR/include" LDFLAGS="-L$OPENBLAS_DIR/lib -lopenblas"
```

### Static Linking

For portable binaries:
```bash
make LDFLAGS="-static -lm"
# Note: May need static versions of libraries
```

### Cross-Compilation

Example for ARM64:
```bash
make CC=aarch64-linux-gnu-gcc
```

### Sanitizers (Development)

```bash
# Address sanitizer
make debug CFLAGS="-fsanitize=address -g"

# Undefined behavior sanitizer
make debug CFLAGS="-fsanitize=undefined -g"
```

---

## Makefile Reference

### Complete Makefile Structure

```makefile
# Configuration
CC ?= gcc
TARGET ?= gemma3
BUILD_DIR ?= build

# Source files
SRCS_BASE = gemma3.c gemma3_kernels.c gemma3_safetensors.c \
            gemma3_tokenizer.c gemma3_transformer.c main.c

# Flag presets
CFLAGS_BASE    = -Wall -Wextra -Wpedantic -std=c11 -MMD -MP
CFLAGS_RELEASE = -O3 -DNDEBUG
CFLAGS_DEBUG   = -g -O0 -DDEBUG
CFLAGS_FAST    = -O3 -march=native -ffast-math -DNDEBUG

LDFLAGS_BASE   = -lm

# Mode-specific configuration
MODE ?= release
# ... (see full Makefile for details)

# Targets
all: release
debug: ...
fast: ...
blas: ...
threads: ...
blas-threads: ...
clean: ...
```

---

## Related Documentation

- [DOCUMENTATION.md](DOCUMENTATION.md) - Main project documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - Architecture deep dive
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API reference
- [INTERNALS.md](INTERNALS.md) - Implementation deep dive
