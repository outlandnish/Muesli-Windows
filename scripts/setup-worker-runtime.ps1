param(
    [switch]$WithPostProcessing,
    [switch]$WithParakeet,
    [switch]$WithParakeetNpu,
    [switch]$WithNpuSummary,
    [switch]$WithDiarization,
    [switch]$WithDiarizeSherpa,
    [switch]$WithDiarizeSortformerNpu,
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (Test-Path (Join-Path $root "Muesli.exe")) {
    $appDir = $root
} else {
    $appDir = Resolve-Path (Join-Path $PSScriptRoot "..\publish\muesli-windows-win-x64")
}

$requirements = Join-Path $appDir "worker\requirements.txt"
$postRequirements = Join-Path $appDir "worker\requirements-postprocess.txt"
$parakeetRequirements = Join-Path $appDir "worker\requirements-parakeet.txt"
$parakeetNpuRequirements = Join-Path $appDir "worker\requirements-parakeet-npu.txt"
$npuSummaryRequirements = Join-Path $appDir "worker\requirements-npu-summary.txt"
$diarizationRequirements = Join-Path $appDir "worker\requirements-diarization.txt"
$diarizeSherpaRequirements = Join-Path $appDir "worker\requirements-diarize-sherpa.txt"
$diarizeSortformerNpuRequirements = Join-Path $appDir "worker\requirements-diarize-sortformer-npu.txt"
# The self-built sherpa-onnx arm64 wheel (no PyPI arm64 wheel exists), bundled
# under worker\wheels\ by package-windows-v1.ps1's arm64 path.
$sherpaWheelDir = Join-Path $appDir "worker\wheels"

if (-not (Test-Path $requirements)) {
    throw "Could not find worker requirements at $requirements"
}

$bundledPython = Join-Path $appDir "python\python.exe"
$useBundled = Test-Path $bundledPython

if ($useBundled) {
    $siteTarget = Join-Path $appDir "python\site-packages-muesli"

    if ($CheckOnly) {
        # What matters for the bundled runtime is that it launches and is CPython
        # 3.x; the exact minor version differs by arch (x64 and arm64 bundles pin
        # different python-build-standalone releases) and isn't load-bearing.
        & $bundledPython -c "import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "Bundled python at $bundledPython did not launch as a CPython 3.x runtime."
        }
        $reported = & $bundledPython -c "import sys, platform; print(f'{sys.version_info.major}.{sys.version_info.minor} {platform.machine()}')"
        Write-Host "Bundled CPython $($reported.Trim()) runtime check passed at $bundledPython"
        return
    }

    New-Item -ItemType Directory -Force -Path $siteTarget | Out-Null

    if (-not (Test-Path (Join-Path $siteTarget "faster_whisper")) -and -not (Test-Path (Join-Path $siteTarget "faster_whisper-*.dist-info"))) {
        Write-Host "Installing base worker dependencies into bundled site-packages"
        & $bundledPython -m pip install --upgrade pip --no-warn-script-location
        if ($LASTEXITCODE -ne 0) { throw "pip self-upgrade failed (exit $LASTEXITCODE)." }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $requirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of base worker requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithPostProcessing) {
        if (-not (Test-Path $postRequirements)) {
            throw "Could not find post-processing requirements at $postRequirements"
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $postRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of Qwen post-processing requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithParakeet) {
        if (-not (Test-Path $parakeetRequirements)) {
            throw "Could not find Parakeet requirements at $parakeetRequirements"
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $parakeetRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of Parakeet requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithParakeetNpu) {
        if (-not (Test-Path $parakeetNpuRequirements)) {
            throw "Could not find Parakeet NPU requirements at $parakeetNpuRequirements"
        }
        # onnxruntime-qnn ships arm64-only wheels; guard against installing into
        # a non-arm64 bundle where the QNN DLLs can't load (Error 193).
        & $bundledPython -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "The parakeet-npu engine requires a native ARM64 Python (Snapdragon). Bundled python at $bundledPython is not arm64."
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $parakeetNpuRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of Parakeet NPU requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithNpuSummary) {
        if (-not (Test-Path $npuSummaryRequirements)) {
            throw "Could not find NPU summary requirements at $npuSummaryRequirements"
        }
        # GenieX ships arm64-only for Windows-on-ARM; require a native ARM64 bundle.
        & $bundledPython -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "The NPU summary (GenieX) engine requires a native ARM64 Python (Snapdragon). Bundled python at $bundledPython is not arm64."
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $npuSummaryRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of NPU summary (GenieX) requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithDiarization) {
        if (-not (Test-Path $diarizationRequirements)) {
            throw "Could not find diarization requirements at $diarizationRequirements"
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $diarizationRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of diarization requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithDiarizeSherpa) {
        if (-not (Test-Path $diarizeSherpaRequirements)) {
            throw "Could not find sherpa diarization requirements at $diarizeSherpaRequirements"
        }
        # sherpa-onnx ships as an arm64-only self-built wheel; require a native
        # ARM64 Python (its .pyd can't load under x64/emulation).
        & $bundledPython -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "The sherpa diarization backend requires a native ARM64 Python (Snapdragon). Bundled python at $bundledPython is not arm64."
        }
        # Install the bundled sherpa-onnx arm64 wheel (no PyPI arm64 wheel exists).
        $sherpaWheel = Get-ChildItem -Path $sherpaWheelDir -Filter "sherpa_onnx-*-win_arm64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $sherpaWheel) {
            throw "Bundled sherpa-onnx arm64 wheel not found under $sherpaWheelDir. The arm64 package must ship it."
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget $sherpaWheel.FullName
        if ($LASTEXITCODE -ne 0) { throw "pip install of bundled sherpa-onnx wheel failed (exit $LASTEXITCODE)." }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $diarizeSherpaRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of sherpa diarization requirements failed (exit $LASTEXITCODE)." }
    }

    if ($WithDiarizeSortformerNpu) {
        if (-not (Test-Path $diarizeSortformerNpuRequirements)) {
            throw "Could not find Sortformer NPU diarization requirements at $diarizeSortformerNpuRequirements"
        }
        # onnxruntime-qnn ships arm64-only wheels and needs a native ARM64 Python to
        # load the QNN HTP libraries (an x64/emulated interpreter cannot).
        & $bundledPython -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
        if ($LASTEXITCODE -ne 0) {
            throw "The Sortformer NPU diarization backend requires a native ARM64 Python (Snapdragon). Bundled python at $bundledPython is not arm64."
        }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $diarizeSortformerNpuRequirements
        if ($LASTEXITCODE -ne 0) { throw "pip install of Sortformer NPU diarization requirements failed (exit $LASTEXITCODE)." }
    }

    Write-Host "Muesli worker runtime is ready (bundled CPython) at $bundledPython"
    return
}

$venv = Join-Path $appDir ".venv"
$python = Join-Path $venv "Scripts\python.exe"

function Find-PythonLauncher {
    $candidates = @(
        @{ File = "py"; Args = @("-3.12") },
        @{ File = "py"; Args = @("-3.11") },
        @{ File = "python"; Args = @() },
        @{ File = "python3"; Args = @() }
    )

    foreach ($candidate in $candidates) {
        try {
            if (-not (Get-Command $candidate.File -ErrorAction SilentlyContinue)) {
                continue
            }
            $versionArgs = @()
            $versionArgs += $candidate.Args
            $versionArgs += @("-c", "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3, 11), (3, 12)] else 1)")
            & $candidate.File @versionArgs | Out-Null
            if ($LASTEXITCODE -eq 0) {
                return $candidate
            }
        } catch {
        }
    }

    throw "Supported Python was not found. Install Python 3.11 or 3.12, then run setup-worker-runtime.ps1 again. Python 3.13+ is not used for this v1 package."
}

if (-not (Test-Path $python)) {
    if ($CheckOnly) {
        $launcher = Find-PythonLauncher
        Write-Host "Supported Python launcher found: $($launcher.File) $($launcher.Args -join ' ')"
        return
    }

    $launcher = Find-PythonLauncher
    $venvArgs = @()
    $venvArgs += $launcher.Args
    $venvArgs += @("-m", "venv", $venv)
    & $launcher.File @venvArgs
}

if ($CheckOnly) {
    & $python -c "import sys; raise SystemExit(0 if sys.version_info[:2] in [(3, 11), (3, 12)] else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "Existing worker runtime uses an unsupported Python version. Delete .venv and rerun setup with Python 3.11 or 3.12."
    }

    Write-Host "Muesli worker runtime check passed at $python"
    return
}

& $python -m pip install --upgrade pip
& $python -m pip install -r $requirements

if ($WithPostProcessing) {
    if (-not (Test-Path $postRequirements)) {
        throw "Could not find post-processing requirements at $postRequirements"
    }
    & $python -m pip install -r $postRequirements
}

if ($WithParakeet) {
    if (-not (Test-Path $parakeetRequirements)) {
        throw "Could not find Parakeet requirements at $parakeetRequirements"
    }
    & $python -m pip install -r $parakeetRequirements
}

if ($WithParakeetNpu) {
    if (-not (Test-Path $parakeetNpuRequirements)) {
        throw "Could not find Parakeet NPU requirements at $parakeetNpuRequirements"
    }
    # onnxruntime-qnn ships arm64-only wheels; require a native ARM64 interpreter.
    & $python -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The parakeet-npu engine requires a native ARM64 Python (Snapdragon). $python is not arm64."
    }
    & $python -m pip install -r $parakeetNpuRequirements
}

if ($WithNpuSummary) {
    if (-not (Test-Path $npuSummaryRequirements)) {
        throw "Could not find NPU summary requirements at $npuSummaryRequirements"
    }
    & $python -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The NPU summary (GenieX) engine requires a native ARM64 Python (Snapdragon). $python is not arm64."
    }
    & $python -m pip install -r $npuSummaryRequirements
}

if ($WithDiarization) {
    if (-not (Test-Path $diarizationRequirements)) {
        throw "Could not find diarization requirements at $diarizationRequirements"
    }
    & $python -m pip install -r $diarizationRequirements
}

if ($WithDiarizeSherpa) {
    if (-not (Test-Path $diarizeSherpaRequirements)) {
        throw "Could not find sherpa diarization requirements at $diarizeSherpaRequirements"
    }
    # sherpa-onnx is a self-built arm64-only wheel; require a native ARM64 Python.
    & $python -c "import sys, struct; f=open(sys.executable,'rb'); f.seek(0x3c); pe=struct.unpack('<I',f.read(4))[0]; f.seek(pe+4); raise SystemExit(0 if struct.unpack('<H',f.read(2))[0]==0xAA64 else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "The sherpa diarization backend requires a native ARM64 Python (Snapdragon). $python is not arm64."
    }
    $sherpaWheel = Get-ChildItem -Path $sherpaWheelDir -Filter "sherpa_onnx-*-win_arm64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $sherpaWheel) {
        throw "Bundled sherpa-onnx arm64 wheel not found under $sherpaWheelDir."
    }
    & $python -m pip install $sherpaWheel.FullName
    & $python -m pip install -r $diarizeSherpaRequirements
}

Write-Host "Muesli worker runtime is ready at $python"
