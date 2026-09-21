# =============================================================
# Build xiaobai_core Rust extension and copy to Python package
# -------------------------------------------------------------
# .\build_rust_core.ps1 [-Release] [-SkipCopy]
# =============================================================
[CmdletBinding()]
param(
    [switch]$Release = $true,
    [switch]$SkipCopy = $false
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$RustDir = Join-Path $Root "xiaobai_core"

if (!(Test-Path (Join-Path $RustDir "Cargo.toml"))) {
    throw "xiaobai_core/Cargo.toml not found at $RustDir"
}

# Check cargo
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (!$cargo) {
    throw "cargo not found. Install Rust: https://rustup.rs/"
}

Write-Host "==> Building xiaobai_core (release=$Release)" -ForegroundColor Cyan
Push-Location $RustDir
try {
    if ($Release) {
        & cargo build --release
    } else {
        & cargo build
    }
    if ($LASTEXITCODE -ne 0) {
        throw "cargo build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

if ($SkipCopy) {
    Write-Host "[skip] Copy to package disabled" -ForegroundColor DarkGray
    return
}

# Copy .dll → .pyd into Python package
$profile = if ($Release) { "release" } else { "debug" }
$DllSrc = Join-Path $RustDir "target\$profile\xiaobai_core.dll"
$PydDst = Join-Path $Root "xiaobai_voice\xiaobai_core.pyd"

if (!(Test-Path $DllSrc)) {
    throw "Built DLL not found: $DllSrc"
}

Copy-Item -Force $DllSrc $PydDst
Write-Host "[OK] Copied xiaobai_core.dll -> xiaobai_voice\xiaobai_core.pyd" -ForegroundColor Green

# Verify import
Write-Host "==> Verifying Python import" -ForegroundColor Cyan
$PyExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($PyExe) {
    Push-Location $Root
    try {
        & $PyExe -c "from xiaobai_voice.core import RUST_AVAILABLE, RUST_VERSION; print(f'Rust core: available={RUST_AVAILABLE}, version={RUST_VERSION}')"
    } finally {
        Pop-Location
    }
}
