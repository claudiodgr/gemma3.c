# gemma3.c - Build Guide

Complete guide for building, configuring, and installing gemma3.c.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Build Systems](#build-systems)
4. [CMake Build (Recommended)](#cmake-build-recommended)
5. [Makefile Build (Legacy)](#makefile-build-legacy)
6. [Platform-Specific Instructions](#platform-specific-instructions)
7. [Model Download](#model-download)
8. [Verification](#verification)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| C Compiler | GCC 7+, Clang 6+, or MSVC 2019+ | C11 compiler |
| CMake | 3.16+ | Build system (recommended) |
| Python 3 | 3.7+ | Model download script |

### Optional Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| OpenBLAS | BLAS acceleration | See platform-specific section |
| wgpu-native | WebGPU GPU acceleration | See [WebGPU Setup](#webgpu-setup) |
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

### CMake (Recommended - Cross-Platform)

```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Or use presets
cmake --preset release
cmake --build --preset release

# Run
./build/gemma3 -m ./gemma-3-4b-it -p "Hello, world!"
```

### Makefile (Linux/macOS)

```bash
make
./gemma3 -m ./gemma-3-4b-it -p "Hello, world!"
```

---

## Build Systems

gemma3.c supports two build systems:

| Build System | Platforms | Recommended For |
|--------------|-----------|-----------------|
| **CMake** | Linux, Windows, macOS | Cross-platform development, IDE integration |
| **Makefile** | Linux, macOS | Quick builds on POSIX systems |

---

## CMake Build (Recommended)

### Basic Usage

```bash
# Configure
cmake -B build

# Build
cmake --build build

# Install (optional)
cmake --install build --prefix /usr/local
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Build type (Debug, Release, RelWithDebInfo) |
| `GEMMA3_USE_THREADS` | OFF | Enable multi-threading support |
| `GEMMA3_USE_BLAS` | OFF | Enable OpenBLAS acceleration |
| `GEMMA3_USE_WEBGPU` | OFF | Enable WebGPU GPU acceleration |
| `GEMMA3_NATIVE` | OFF | Enable native CPU optimizations |
| `GEMMA3_BUILD_SHARED` | OFF | Build shared library instead of static |
| `GEMMA3_BUILD_CLI` | ON | Build command-line interface |

### Example Configurations

```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug

# Release with threading
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGEMMA3_USE_THREADS=ON

# Full optimization with BLAS and threading
cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DGEMMA3_NATIVE=ON \
    -DGEMMA3_USE_BLAS=ON \
    -DGEMMA3_USE_THREADS=ON

# WebGPU GPU acceleration
cmake -B build -DGEMMA3_USE_WEBGPU=ON
```

### CMake Presets

Use presets for convenient configuration:

```bash
# List available presets
cmake --list-presets

# Configure with a preset
cmake --preset release
cmake --preset threads
cmake --preset blas-threads
cmake --preset webgpu
cmake --preset full

# Build with a preset
cmake --build --preset release
```

### Visual Studio (Windows)

```powershell
# Generate Visual Studio solution
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build from command line
cmake --build build --config Release

# Or open build/gemma3.sln in Visual Studio
```

### Xcode (macOS)

```bash
# Generate Xcode project
cmake -B build -G Xcode

# Build from command line
cmake --build build --config Release

# Or open build/gemma3.xcodeproj in Xcode
```

### Ninja (Faster Builds)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Makefile Build (Legacy)

For POSIX systems (Linux, macOS) with Make installed.

### Available Targets

| Target | Command | Description |
|--------|---------|-------------|
| Release | `make` | Default optimized build |
| Debug | `make debug` | Debug symbols, no optimization |
| Fast | `make fast` | Native CPU optimizations |
| Threads | `make threads` | Multi-threaded inference |
| BLAS | `make blas` | OpenBLAS acceleration |
| BLAS+Threads | `make blas-threads` | Best CPU performance |
| WebGPU | `make webgpu` | GPU acceleration |
| WebGPU+Threads | `make webgpu-threads` | GPU + CPU threading |
| Clean | `make clean` | Remove build artifacts |
| Help | `make help` | Show available targets |

---

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install build tools
sudo apt update
sudo apt install build-essential cmake

# Optional: OpenBLAS
sudo apt install libopenblas-dev

# Optional: Vulkan (for WebGPU)
sudo apt install libvulkan-dev vulkan-tools

# Build with CMake
cmake -B build -DGEMMA3_USE_THREADS=ON -DGEMMA3_USE_BLAS=ON
cmake --build build
```

### Linux (Fedora/RHEL)

```bash
# Install build tools
sudo dnf groupinstall "Development Tools"
sudo dnf install cmake

# Optional: OpenBLAS
sudo dnf install openblas-devel

# Build
cmake -B build -DGEMMA3_USE_THREADS=ON -DGEMMA3_USE_BLAS=ON
cmake --build build
```

### macOS

```bash
# Install Xcode command line tools
xcode-select --install

# Install CMake (via Homebrew)
brew install cmake

# Optional: OpenBLAS
brew install openblas

# Build (specify OpenBLAS path if needed)
cmake -B build \
    -DGEMMA3_USE_THREADS=ON \
    -DGEMMA3_USE_BLAS=ON \
    -DOpenBLAS_ROOT=/opt/homebrew/opt/openblas
cmake --build build
```

### Windows (Visual Studio)

```powershell
# Install Visual Studio 2019 or later with C++ workload
# Install CMake (download from cmake.org or use winget)
winget install Kitware.CMake

# Configure
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build
cmake --build build --config Release

# Or open build\gemma3.sln in Visual Studio
```

### Windows (MinGW-w64)

```bash
# Install MSYS2 from https://www.msys2.org/
# In MSYS2 terminal:
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake

# Build
cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Windows (WSL)

```bash
# Inside WSL (Ubuntu)
sudo apt update
sudo apt install build-essential cmake

# Follow Linux instructions
cmake -B build -DGEMMA3_USE_THREADS=ON
cmake --build build
```

---

## WebGPU Setup

### Download wgpu-native Libraries

**Automatic download:**
```bash
# Linux/macOS
./scripts/download_webgpu.sh

# Windows (PowerShell)
.\scripts\download_webgpu.ps1
```

**Manual download:**
1. Go to https://github.com/gfx-rs/wgpu-native/releases
2. Download the appropriate archive for your platform
3. Extract to `lib/webgpu/<platform>/`

See `lib/webgpu/README.md` for detailed instructions.

### Build with WebGPU

```bash
cmake -B build -DGEMMA3_USE_WEBGPU=ON
cmake --build build
```

### GPU Driver Requirements

- **Linux**: Vulkan drivers (Mesa, NVIDIA, AMD)
- **Windows**: Direct3D 12 or Vulkan drivers
- **macOS**: Metal (built into macOS 10.14+)

---

## Model Download

### Using download_model.py

```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here  # Linux/macOS
$env:HF_TOKEN = "your_token_here"  # Windows PowerShell

# Install huggingface_hub
pip install huggingface_hub

# Download model
python download_model.py

# Model files will be in ./gemma-3-4b-it/
```

### Required Files

| File | Size | Description |
|------|------|-------------|
| `model-00001-of-00002.safetensors` | ~4 GB | Model weights part 1 |
| `model-00002-of-00002.safetensors` | ~4 GB | Model weights part 2 |
| `tokenizer.model` | ~4 MB | SentencePiece tokenizer |

---

## Verification

### Check Build

```bash
# Show version/help
./build/gemma3 --help

# Test with a simple prompt
./build/gemma3 -m ./gemma-3-4b-it -p "What is 2+2?" -n 50
```

### Check GPU (WebGPU build)

```bash
# The binary will report GPU detection on startup
./build/gemma3 -m ./gemma-3-4b-it -p "Hello" -v
```

---

## Troubleshooting

### CMake Errors

**"Could not find OpenBLAS"**
```bash
# Linux
sudo apt install libopenblas-dev

# macOS
brew install openblas
cmake -B build -DOpenBLAS_ROOT=/opt/homebrew/opt/openblas

# Windows: Download OpenBLAS and set path
cmake -B build -DOpenBLAS_ROOT=C:/OpenBLAS
```

**"Could not find WebGPU"**
```bash
# Download wgpu-native libraries first
./scripts/download_webgpu.sh  # or .ps1 on Windows

# Or specify path manually
cmake -B build -DWEBGPU_DIR=/path/to/wgpu
```

### Runtime Errors

**"Failed to load model"**
- Ensure all `.safetensors` files are present
- Check file permissions
- Verify the model path is correct

**"Out of memory"**
- Reduce context size: `./gemma3 -m model -c 512`
- Use a machine with more RAM
- Check for memory leaks with valgrind

**"mmap failed"**
- Insufficient virtual address space
- Try reducing context size
- On 32-bit systems, use smaller models

### Performance Issues

**Slow inference**
```bash
# Use optimized build
cmake -B build -DGEMMA3_NATIVE=ON -DGEMMA3_USE_THREADS=ON -DGEMMA3_USE_BLAS=ON
cmake --build build

# Or use WebGPU for GPU acceleration
cmake -B build -DGEMMA3_USE_WEBGPU=ON
```

### Build Performance

**Slow compilation**
```bash
# Use Ninja for faster builds
cmake -B build -G Ninja
cmake --build build

# Use parallel compilation
cmake --build build -j$(nproc)  # Linux
cmake --build build -j$(sysctl -n hw.ncpu)  # macOS
```

---

## IDE Integration

### VS Code

Install the CMake Tools extension, then:
1. Open the project folder
2. Press Ctrl+Shift+P → "CMake: Configure"
3. Select a preset or configure manually
4. Press Ctrl+Shift+P → "CMake: Build"

### CLion

1. Open the project folder (CMakeLists.txt will be detected)
2. Configure CMake options in Settings → Build → CMake
3. Build with Ctrl+F9

### Visual Studio

1. Open the folder or generated .sln file
2. Select configuration (Debug/Release)
3. Build with Ctrl+Shift+B
