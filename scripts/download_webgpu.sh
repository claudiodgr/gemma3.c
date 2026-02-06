#!/bin/bash
# Download wgpu-native libraries for the current platform
# Usage: ./scripts/download_webgpu.sh [--all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_ROOT/lib/webgpu"

# wgpu-native release version
WGPU_VERSION="v0.19.4.1"
WGPU_BASE_URL="https://github.com/gfx-rs/wgpu-native/releases/download/${WGPU_VERSION}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

download_and_extract() {
    local platform=$1
    local archive=$2
    local dest_dir="$LIB_DIR/$platform"

    log_info "Downloading $archive..."
    local url="${WGPU_BASE_URL}/${archive}"
    local tmp_file="/tmp/${archive}"

    if command -v curl &> /dev/null; then
        curl -L -o "$tmp_file" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$tmp_file" "$url"
    else
        log_error "Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    log_info "Extracting to $dest_dir..."
    mkdir -p "$dest_dir"

    if [[ "$archive" == *.zip ]]; then
        unzip -o "$tmp_file" -d "$dest_dir"
    elif [[ "$archive" == *.tar.gz ]]; then
        tar -xzf "$tmp_file" -C "$dest_dir"
    fi

    rm -f "$tmp_file"
    log_info "Done: $platform"
}

detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Linux)
            case "$arch" in
                x86_64) echo "linux-x86_64" ;;
                aarch64) echo "linux-aarch64" ;;
                *) log_error "Unsupported Linux architecture: $arch"; exit 1 ;;
            esac
            ;;
        Darwin)
            case "$arch" in
                x86_64) echo "macos-x86_64" ;;
                arm64) echo "macos-arm64" ;;
                *) log_error "Unsupported macOS architecture: $arch"; exit 1 ;;
            esac
            ;;
        *)
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
}

get_archive_name() {
    local platform=$1
    case "$platform" in
        linux-x86_64) echo "wgpu-linux-x86_64-release.zip" ;;
        linux-aarch64) echo "wgpu-linux-aarch64-release.zip" ;;
        macos-x86_64) echo "wgpu-macos-x86_64-release.zip" ;;
        macos-arm64) echo "wgpu-macos-aarch64-release.zip" ;;
        windows-x86_64) echo "wgpu-windows-x86_64-msvc-release.zip" ;;
        *) log_error "Unknown platform: $platform"; exit 1 ;;
    esac
}

main() {
    log_info "wgpu-native downloader (version: $WGPU_VERSION)"

    if [[ "$1" == "--all" ]]; then
        log_info "Downloading all platforms..."
        for platform in linux-x86_64 macos-x86_64 macos-arm64 windows-x86_64; do
            archive=$(get_archive_name "$platform")
            download_and_extract "$platform" "$archive"
        done
    else
        local platform=$(detect_platform)
        log_info "Detected platform: $platform"
        local archive=$(get_archive_name "$platform")
        download_and_extract "$platform" "$archive"
    fi

    log_info "All downloads complete!"
    log_info "Configure with: cmake -B build -DGEMMA3_USE_WEBGPU=ON"
}

main "$@"
