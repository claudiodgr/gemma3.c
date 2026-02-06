# Download wgpu-native libraries for Windows
# Usage: .\scripts\download_webgpu.ps1 [-All]

param(
    [switch]$All
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$LibDir = Join-Path $ProjectRoot "lib\webgpu"

# wgpu-native release version
$WgpuVersion = "v27.0.4.0"
$WgpuBaseUrl = "https://github.com/gfx-rs/wgpu-native/releases/download/$WgpuVersion"

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Download-AndExtract {
    param(
        [string]$Platform,
        [string]$Archive
    )

    $DestDir = Join-Path $LibDir $Platform
    $Url = "$WgpuBaseUrl/$Archive"
    $TempFile = Join-Path $env:TEMP $Archive

    Write-Info "Downloading $Archive..."
    Invoke-WebRequest -Uri $Url -OutFile $TempFile -UseBasicParsing

    Write-Info "Extracting to $DestDir..."
    New-Item -ItemType Directory -Force -Path $DestDir | Out-Null

    Expand-Archive -Path $TempFile -DestinationPath $DestDir -Force

    Remove-Item $TempFile -Force
    Write-Info "Done: $Platform"
}

function Get-ArchiveName {
    param([string]$Platform)

    switch ($Platform) {
        "windows-x86_64" { return "wgpu-windows-x86_64-msvc-release.zip" }
        "linux-x86_64" { return "wgpu-linux-x86_64-release.zip" }
        "macos-x86_64" { return "wgpu-macos-x86_64-release.zip" }
        "macos-arm64" { return "wgpu-macos-aarch64-release.zip" }
        default { throw "Unknown platform: $Platform" }
    }
}

function Main {
    Write-Info "wgpu-native downloader (version: $WgpuVersion)"

    if ($All) {
        Write-Info "Downloading all platforms..."
        $Platforms = @("windows-x86_64", "linux-x86_64", "macos-x86_64", "macos-arm64")
        foreach ($Platform in $Platforms) {
            $Archive = Get-ArchiveName $Platform
            Download-AndExtract -Platform $Platform -Archive $Archive
        }
    }
    else {
        # Detect current platform
        $Platform = "windows-x86_64"
        if ([Environment]::Is64BitOperatingSystem -eq $false) {
            throw "Only 64-bit Windows is supported"
        }

        Write-Info "Detected platform: $Platform"
        $Archive = Get-ArchiveName $Platform
        Download-AndExtract -Platform $Platform -Archive $Archive
    }

    Write-Info "All downloads complete!"
    Write-Info "Configure with: cmake -B build -DGEMMA3_USE_WEBGPU=ON"
}

Main
