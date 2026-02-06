# WebGPU Native Libraries

This directory contains platform-specific wgpu-native libraries for GPU acceleration.

## Directory Structure

```
lib/webgpu/
├── linux-x86_64/
│   ├── include/webgpu/webgpu.h
│   └── lib/libwgpu_native.a (or .so)
├── windows-x86_64/
│   ├── include/webgpu/webgpu.h
│   ├── lib/wgpu_native.lib
│   └── bin/wgpu_native.dll
├── macos-x86_64/
│   ├── include/webgpu/webgpu.h
│   └── lib/libwgpu_native.a (or .dylib)
└── macos-arm64/
    ├── include/webgpu/webgpu.h
    └── lib/libwgpu_native.a (or .dylib)
```

## Download Instructions

### Automatic Download (Recommended)

Run the download script from the project root:

```bash
# Linux/macOS
./scripts/download_webgpu.sh

# Windows (PowerShell)
.\scripts\download_webgpu.ps1
```

### Manual Download

1. Go to the wgpu-native releases page:
   https://github.com/gfx-rs/wgpu-native/releases

2. Download the appropriate archive for your platform:
   - **Linux x86_64**: `wgpu-linux-x86_64-release.zip`
   - **Windows x86_64**: `wgpu-windows-x86_64-release.zip`
   - **macOS x86_64**: `wgpu-macos-x86_64-release.zip`
   - **macOS ARM64**: `wgpu-macos-aarch64-release.zip`

3. Extract the archive and copy files:

   **Linux:**
   ```bash
   unzip wgpu-linux-x86_64-release.zip -d lib/webgpu/linux-x86_64/
   ```

   **Windows:**
   ```powershell
   Expand-Archive wgpu-windows-x86_64-release.zip -DestinationPath lib\webgpu\windows-x86_64\
   ```

   **macOS (Intel):**
   ```bash
   unzip wgpu-macos-x86_64-release.zip -d lib/webgpu/macos-x86_64/
   ```

   **macOS (Apple Silicon):**
   ```bash
   unzip wgpu-macos-aarch64-release.zip -d lib/webgpu/macos-arm64/
   ```

## System Installation (Alternative)

Instead of placing libraries here, you can install wgpu-native system-wide:

### Linux
```bash
sudo cp lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
sudo ldconfig
```

### macOS (Homebrew)
```bash
# If available via Homebrew
brew install wgpu-native

# Or manually
sudo cp lib/* /usr/local/lib/
sudo cp -r include/* /usr/local/include/
```

### Windows
Add the library directory to your PATH or copy to a system location.

## Verifying Installation

After placing the libraries, configure with WebGPU enabled:

```bash
cmake -B build -DGEMMA3_USE_WEBGPU=ON
```

If CMake finds the library, you'll see:
```
-- WebGPU: Found at <path>
```

## GPU Driver Requirements

WebGPU requires appropriate GPU drivers:

- **Linux**: Vulkan drivers (Mesa, NVIDIA, AMD)
- **Windows**: Direct3D 12 or Vulkan drivers
- **macOS**: Metal (built into macOS 10.14+)

Check GPU support:
```bash
# Linux
vulkaninfo | head

# Windows (PowerShell)
dxdiag

# macOS
system_profiler SPDisplaysDataType
```
