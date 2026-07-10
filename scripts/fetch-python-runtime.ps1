param(
    [Parameter(Mandatory = $true)]
    [string]$DestinationParent,
    [ValidateSet("x64", "arm64")]
    [string]$Arch = "x64",
    [string]$Version = "",
    [string]$BuildStamp = "",
    [string]$ExpectedSha256 = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DestinationParent)) {
    throw "DestinationParent does not exist: $DestinationParent"
}

# Per-arch defaults. The x64 runtime is CPython 3.12 (the shipped v1 baseline).
# The arm64 runtime is CPython 3.13, required by the onnxruntime-qnn wheel used by
# the parakeet-v3-npu engine (Snapdragon NPU). python-build-standalone only began
# publishing aarch64-pc-windows-msvc in later releases, so the two arches pin
# different build stamps. All are install_only tarballs, SHA-256 pinned.
$archDefaults = @{
    "x64"   = @{
        Triple = "x86_64-pc-windows-msvc"
        Version = "3.12.11"
        BuildStamp = "20250612"
        Sha256 = "7b93afa91931dbc37b307a81b8680b30193736b5ef29a44ef6452f702c306e7a"
    }
    "arm64" = @{
        Triple = "aarch64-pc-windows-msvc"
        Version = "3.13.14"
        BuildStamp = "20260623"
        Sha256 = "d425c0fa7fe5e9889760b199c87e45cb84d793639ed63231f9313ffa894cb25a"
    }
}
$defaults = $archDefaults[$Arch]
if ([string]::IsNullOrWhiteSpace($Version)) { $Version = $defaults.Version }
if ([string]::IsNullOrWhiteSpace($BuildStamp)) { $BuildStamp = $defaults.BuildStamp }
if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) { $ExpectedSha256 = $defaults.Sha256 }
if ([string]::IsNullOrWhiteSpace($ExpectedSha256)) {
    throw "No pinned SHA-256 for the $Arch CPython $Version tarball. Pass -ExpectedSha256 or fill in archDefaults['$Arch'].Sha256."
}

$assetName = "cpython-$Version+$BuildStamp-$($defaults.Triple)-install_only.tar.gz"
$downloadUrl = "https://github.com/astral-sh/python-build-standalone/releases/download/$BuildStamp/$assetName"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$cacheDir = Join-Path $repoRoot "artifacts\python-cache"
New-Item -ItemType Directory -Force -Path $cacheDir | Out-Null
$cachedTarball = Join-Path $cacheDir $assetName

function Get-Sha256([string]$path) {
    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$expected = $ExpectedSha256.ToLowerInvariant()

if ((Test-Path $cachedTarball) -and -not $Force) {
    $actual = Get-Sha256 $cachedTarball
    if ($actual -ne $expected) {
        Write-Host "Cached tarball SHA256 mismatch (expected $expected, got $actual). Re-downloading."
        Remove-Item -LiteralPath $cachedTarball -Force
    }
}

if (-not (Test-Path $cachedTarball)) {
    Write-Host "Downloading $assetName"
    $previous = $ProgressPreference
    $ProgressPreference = "SilentlyContinue"
    try {
        Invoke-WebRequest -UseBasicParsing -Uri $downloadUrl -OutFile $cachedTarball
    } finally {
        $ProgressPreference = $previous
    }
    $actual = Get-Sha256 $cachedTarball
    if ($actual -ne $expected) {
        Remove-Item -LiteralPath $cachedTarball -Force -ErrorAction SilentlyContinue
        throw "SHA256 mismatch for $assetName. Expected $expected, got $actual."
    }
}

$pythonDir = Join-Path $DestinationParent "python"
if (Test-Path $pythonDir) {
    if (-not $Force) {
        # Reuse existing extraction only if the bundled python.exe runs and reports the expected version.
        $existingPython = Join-Path $pythonDir "python.exe"
        if (Test-Path $existingPython) {
            try {
                $reported = & $existingPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
                if ($LASTEXITCODE -eq 0 -and $reported.Trim() -eq $Version) {
                    Write-Host "Bundled CPython $Version already in place at $pythonDir"
                    return
                }
            } catch {
            }
        }
    }
    Remove-Item -LiteralPath $pythonDir -Recurse -Force
}

Write-Host "Unpacking $assetName -> $DestinationParent"
$tarExe = Join-Path $env:SystemRoot "System32\tar.exe"
if (-not (Test-Path $tarExe)) {
    $tarExe = "tar"
}
& $tarExe -xzf $cachedTarball -C $DestinationParent
if ($LASTEXITCODE -ne 0) {
    throw "tar exit code $LASTEXITCODE while unpacking $assetName"
}

$bundledPython = Join-Path $pythonDir "python.exe"
if (-not (Test-Path $bundledPython)) {
    throw "Expected $bundledPython after unpacking, but it does not exist."
}

$license = Join-Path $pythonDir "LICENSE.txt"
if (Test-Path $license) {
    Copy-Item -LiteralPath $license -Destination (Join-Path $DestinationParent "THIRD-PARTY-NOTICES-PYTHON.txt") -Force
}

Write-Host "Bundled CPython $Version at $pythonDir"
