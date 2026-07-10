param(
    [string]$Configuration = "Release",
    [string]$Runtime = "win-x64",
    [string]$SentryDsn = "",
    # Path to the self-built sherpa-onnx arm64 wheel to bundle for CPU speaker
    # diarization (no PyPI arm64 wheel exists). Defaults to worker\wheels\ in the
    # repo. Only used for -Runtime win-arm64. Env override: MUESLI_SHERPA_WHEEL.
    [string]$SherpaWheel = "",
    # Path to the pre-compiled Sortformer QNN context binary (sortformer.bin) to
    # bundle for NPU speaker diarization. Only used for -Runtime win-arm64. If unset,
    # the bundle ships without it and diarization uses the sherpa CPU fallback until
    # the .bin is added. Env override: MUESLI_SORTFORMER_BIN.
    [string]$SortformerBin = ""
)

$ErrorActionPreference = "Stop"

# The bundled-Python architecture follows the publish runtime. arm64 additionally
# gets the parakeet-npu (Snapdragon Hexagon NPU) worker dependencies.
switch ($Runtime) {
    "win-x64"   { $pythonArch = "x64" }
    "win-arm64" { $pythonArch = "arm64" }
    default     { throw "Unsupported -Runtime '$Runtime' (expected win-x64 or win-arm64)." }
}

if ([string]::IsNullOrWhiteSpace($SentryDsn) -and -not [string]::IsNullOrWhiteSpace($env:SENTRY_DSN)) {
    $SentryDsn = $env:SENTRY_DSN
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$project = Join-Path $root "windows-native\Muesli.Windows\Muesli.Windows.csproj"
$publishRoot = Join-Path $root "publish"
$publishDir = Join-Path $publishRoot "muesli-windows-$Runtime"
$artifactsDir = Join-Path $root "artifacts"
$zipPath = Join-Path $artifactsDir "muesli-windows-v1-$Runtime.zip"
$lastPublishFile = Join-Path $artifactsDir "last-publish-dir.txt"

if (Test-Path $publishDir) {
    Get-Process Muesli -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    for ($attempt = 1; $attempt -le 5; $attempt++) {
        try {
            Remove-Item -LiteralPath $publishDir -Recurse -Force
            break
        } catch {
            if ($attempt -eq 5) {
                $timestamp = Get-Date -Format "yyyyMMddHHmmss"
                $publishDir = Join-Path $publishRoot "muesli-windows-$Runtime-$timestamp"
                Write-Warning "Could not clean existing publish directory. Publishing to '$publishDir' instead. Last error: $($_.Exception.Message)"
                break
            }
            Start-Sleep -Milliseconds (350 * $attempt)
        }
    }
}
New-Item -ItemType Directory -Force -Path $publishDir | Out-Null
New-Item -ItemType Directory -Force -Path $artifactsDir | Out-Null

$publishArgs = @(
    $project,
    "-c", $Configuration,
    "-r", $Runtime,
    "--self-contained", "true",
    "-p:PublishSingleFile=false",
    "-p:PublishReadyToRun=true",
    "-o", $publishDir
)
if (-not [string]::IsNullOrWhiteSpace($SentryDsn)) {
    $publishArgs += "-p:SentryDsn=$SentryDsn"
    Write-Host "Embedding Sentry DSN into release build."
}
dotnet publish @publishArgs
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed with exit $LASTEXITCODE." }

$satelliteCultureDirs = @(
    "cs", "de", "es", "fr", "it", "ja", "ko", "pl", "pt-BR", "ru", "tr", "zh-Hans", "zh-Hant"
)
foreach ($culture in $satelliteCultureDirs) {
    $cultureDir = Join-Path $publishDir $culture
    if (Test-Path $cultureDir) {
        Remove-Item -LiteralPath $cultureDir -Recurse -Force
    }
}

Get-ChildItem -LiteralPath $publishDir -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $publishDir -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force -ErrorAction SilentlyContinue

& (Join-Path $PSScriptRoot "fetch-python-runtime.ps1") -DestinationParent $publishDir -Arch $pythonArch
if ($LASTEXITCODE -ne 0) {
    throw "fetch-python-runtime.ps1 exited with code $LASTEXITCODE"
}

$bundledPython = Join-Path $publishDir "python\python.exe"
$siteTarget = Join-Path $publishDir "python\site-packages-muesli"

New-Item -ItemType Directory -Force -Path $siteTarget | Out-Null
& $bundledPython -m pip install --upgrade pip --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw "Bundled pip self-upgrade failed (exit $LASTEXITCODE)." }

if ($pythonArch -eq "arm64") {
    # Snapdragon: Whisper's backend (ctranslate2) has NO win_arm64 wheel, so the
    # x64 base requirements can't install here. The on-device ASR engine on arm64
    # is parakeet-v3-npu (onnxruntime-qnn + onnx-asr, arm64-only wheels) — install
    # that as the base, and the GenieX NPU summary deps. Whisper is unavailable on
    # this build by design; the app defaults to the NPU engine.
    foreach ($reqName in @("requirements-parakeet-npu.txt", "requirements-npu-summary.txt")) {
        $req = Join-Path $publishDir "worker\$reqName"
        if (-not (Test-Path $req)) { throw "Requirements file missing at $req after publish." }
        & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $req
        if ($LASTEXITCODE -ne 0) { throw "Bundled pip install of $reqName failed (exit $LASTEXITCODE)." }
    }

    # CPU speaker diarization via sherpa-onnx. There is no PyPI win_arm64 wheel, so
    # we bundle a self-built one under worker\wheels\ and install it here (so the
    # arm64 bundle ships diarization ready to use, like parakeet-npu above). The
    # wheel is also kept in the package for setup-worker-runtime.ps1 -WithDiarizeSherpa.
    $sherpaSrc = $SherpaWheel
    if ([string]::IsNullOrWhiteSpace($sherpaSrc)) { $sherpaSrc = $env:MUESLI_SHERPA_WHEEL }
    if ([string]::IsNullOrWhiteSpace($sherpaSrc)) {
        $repoWheelDir = Join-Path $PSScriptRoot "..\worker\wheels"
        $found = Get-ChildItem -Path $repoWheelDir -Filter "sherpa_onnx-*-win_arm64.whl" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { $sherpaSrc = $found.FullName }
    }
    if ([string]::IsNullOrWhiteSpace($sherpaSrc) -or -not (Test-Path $sherpaSrc)) {
        throw "sherpa-onnx arm64 wheel not found. Pass -SherpaWheel <path>, set MUESLI_SHERPA_WHEEL, or place sherpa_onnx-*-win_arm64.whl under worker\wheels\."
    }
    $bundledWheelDir = Join-Path $publishDir "worker\wheels"
    New-Item -ItemType Directory -Force -Path $bundledWheelDir | Out-Null
    Copy-Item -LiteralPath $sherpaSrc -Destination $bundledWheelDir -Force
    & $bundledPython -m pip install --no-warn-script-location --target $siteTarget $sherpaSrc
    if ($LASTEXITCODE -ne 0) { throw "Bundled pip install of sherpa-onnx wheel failed (exit $LASTEXITCODE)." }
    $sherpaReq = Join-Path $publishDir "worker\requirements-diarize-sherpa.txt"
    if (-not (Test-Path $sherpaReq)) { throw "Requirements file missing at $sherpaReq after publish." }
    & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $sherpaReq
    if ($LASTEXITCODE -ne 0) { throw "Bundled pip install of requirements-diarize-sherpa.txt failed (exit $LASTEXITCODE)." }

    # NPU speaker diarization via NVIDIA Sortformer. Deps are onnxruntime-qnn (already
    # installed above for parakeet-npu) + numpy/scipy/huggingface_hub; install the req
    # explicitly so the bundle ships it ready. The pre-compiled QNN context binary
    # (sortformer.bin) is bundled into the model cache under worker\models\ when
    # available (pass -SortformerBin or set MUESLI_SORTFORMER_BIN); the tiny EPContext
    # wrapper is generated on first download_model(kind="diarize-sortformer-npu").
    $sortformerReq = Join-Path $publishDir "worker\requirements-diarize-sortformer-npu.txt"
    if (-not (Test-Path $sortformerReq)) { throw "Requirements file missing at $sortformerReq after publish." }
    & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $sortformerReq
    if ($LASTEXITCODE -ne 0) { throw "Bundled pip install of requirements-diarize-sortformer-npu.txt failed (exit $LASTEXITCODE)." }

    $sortformerBinSrc = $SortformerBin
    if ([string]::IsNullOrWhiteSpace($sortformerBinSrc)) { $sortformerBinSrc = $env:MUESLI_SORTFORMER_BIN }
    if (-not [string]::IsNullOrWhiteSpace($sortformerBinSrc) -and (Test-Path $sortformerBinSrc)) {
        # Bundle the .bin AND its pre-made EPContext wrapper under worker\ (copied into
        # the build); download_model(kind="diarize-sortformer-npu") copies both into
        # cache_dir()/diarize-sortformer/. Shipping the wrapper pre-made means the
        # runtime needs no `onnx` package to build it. The wrapper is expected next to
        # the .bin (sortformer_ctx.onnx); if absent it is generated on first run.
        $sortformerBundleDir = Join-Path $publishDir "worker\sortformer-bin"
        New-Item -ItemType Directory -Force -Path $sortformerBundleDir | Out-Null
        Copy-Item -LiteralPath $sortformerBinSrc -Destination (Join-Path $sortformerBundleDir "sortformer.bin") -Force
        $sortformerWrapSrc = Join-Path (Split-Path -Parent $sortformerBinSrc) "sortformer_ctx.onnx"
        if (Test-Path $sortformerWrapSrc) {
            Copy-Item -LiteralPath $sortformerWrapSrc -Destination (Join-Path $sortformerBundleDir "sortformer_ctx.onnx") -Force
        }
        Write-Host "Bundled Sortformer NPU context binary from $sortformerBinSrc"
    } else {
        Write-Host "NOTE: no Sortformer NPU binary bundled (pass -SortformerBin or set MUESLI_SORTFORMER_BIN). Diarization will use the sherpa CPU fallback until the .bin is present."
    }
} else {
    # x64: the shipped baseline — Whisper worker dependencies.
    $workerRequirements = Join-Path $publishDir "worker\requirements.txt"
    if (-not (Test-Path $workerRequirements)) {
        throw "Worker requirements file missing at $workerRequirements after publish."
    }
    & $bundledPython -m pip install --no-warn-script-location --target $siteTarget -r $workerRequirements
    if ($LASTEXITCODE -ne 0) { throw "Bundled pip install of worker requirements failed (exit $LASTEXITCODE)." }
}

# -ErrorAction SilentlyContinue on the removal too: enumeration and deletion race
# (a .pyc can vanish between listing and Remove-Item), and with $ErrorActionPreference
# = "Stop" an unguarded failure here would abort the whole package. This cleanup is
# best-effort and must never be fatal.
Get-ChildItem -LiteralPath $siteTarget -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $siteTarget -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in @(".pyc", ".pyo") } |
    Remove-Item -Force -ErrorAction SilentlyContinue

$readme = @"
Muesli for Windows v1
=====================

Run:
  Muesli.exe

Install:
  powershell -ExecutionPolicy Bypass -File .\install-windows.ps1
  Optional:
    powershell -ExecutionPolicy Bypass -File .\install-windows.ps1 -StartAtLogin

Current v1 requirements:
  - Windows x64.
  - No system Python required. A bundled CPython 3.12 runtime ships in this folder
    under `python\`, with base Whisper worker dependencies pre-installed in
    `python\site-packages-muesli`.
  - Verify the extracted/installed package shape:
      powershell -ExecutionPolicy Bypass -File .\fresh-machine-qa.ps1

Optional add-ons (download extra dependencies into the bundled runtime):
  - Qwen transcript cleanup:
      powershell -ExecutionPolicy Bypass -File .\setup-worker-runtime.ps1 -WithPostProcessing
      set MUESLI_ALLOW_MODEL_DOWNLOAD=1 for the first Qwen model download, or preinstall the model in `%USERPROFILE%\.cache\muesli`.
  - NVIDIA Parakeet backend (CUDA/NVIDIA machines):
      powershell -ExecutionPolicy Bypass -File .\setup-worker-runtime.ps1 -WithParakeet
  - Speaker diarization (pyannote, x64/CUDA):
      powershell -ExecutionPolicy Bypass -File .\setup-worker-runtime.ps1 -WithDiarization
  - Speaker diarization (sherpa-onnx CPU, Snapdragon/arm64 — pre-installed in the
    arm64 package; this re-installs from the bundled wheel if needed):
      powershell -ExecutionPolicy Bypass -File .\setup-worker-runtime.ps1 -WithDiarizeSherpa

Default shortcut:
  Hold the configured shortcut to dictate. Release it to transcribe and paste into the previously focused app.
  The default is F8, and it can be changed in Settings.

Notes:
  - This build is local-first and uses Whisper through the bundled worker script.
  - NVIDIA Parakeet v3 is selectable as an optional backend when the Parakeet runtime dependencies are installed.
  - Optional Qwen post-processing can clean grammar, punctuation, and dictated lists after transcription.
  - The Models page can download the selected Whisper model and the optional Qwen cleanup model into the local cache.
  - Meeting summaries can use local fallback, OpenAI, or OpenRouter. Enter provider keys in Settings, or set OPENAI_API_KEY / OPENROUTER_API_KEY.
  - Model cache is stored in `%USERPROFILE%\.cache\muesli`.
  - The Models page includes runtime diagnostics, model cache status, and cache management.
  - Use Models > First-run Setup to check worker readiness, cache the selected Whisper model, open logs, or open the model cache.
  - Settings includes "Start Muesli when I sign in"; it launches the app in the background tray using --background.
  - Settings are stored in `%APPDATA%\muesli\windows-settings.json`.
  - Dictations, meetings, and dictionary data are stored in `%APPDATA%\muesli\data`.
  - Logs are stored in `%APPDATA%\muesli\logs`; use About > Open Logs when reporting issues.
"@

Set-Content -LiteralPath (Join-Path $publishDir "README-WINDOWS.txt") -Value $readme -Encoding UTF8
$releaseNotes = @"
Muesli Windows v0.2.0 Release Notes
===================================

This build is a native Windows WPF clone of the shipped macOS Muesli app.

Highlights:
  - Local-first dictation with Faster-Whisper / Whisper.
  - Push-to-talk and captured custom shortcuts, including alternatives when F8 is taken.
  - Active-app paste after dictation.
  - Meeting detection prompts for Google Meet, Zoom, Teams, and Webex foreground windows.
  - Meeting recording, import, notes, transcripts, folders, and search.
  - First-run onboarding for microphone, shortcut, startup, indicator, and model readiness.
  - Optional NVIDIA Parakeet backend path.
  - Optional Qwen transcript cleanup.

Known release requirements:
  - No system Python required. CPython 3.12 ships bundled in this package.
  - Qwen cleanup requires optional post-processing dependencies and a downloaded local model.
  - Parakeet requires NVIDIA/CUDA-capable hardware and optional Parakeet dependencies.
  - Installer is not code-signed until a real signing certificate is configured.

Fresh-machine validation checklist:
  1. Install using Muesli-win-Setup.exe from artifacts\velopack.
  2. Confirm onboarding appears once.
  3. Pick a microphone and a shortcut that is not reserved by the test laptop.
  4. Run Models > First-run Setup > Check setup.
  5. Download Whisper base and verify dictation with paste into Notepad and Chrome.
  6. Reboot if Launch at login is enabled and confirm Muesli starts in the tray/background.
  7. Test Google Meet in Chrome and Zoom desktop meeting detection prompts.
  8. Test recorded meeting transcript and Notes/Transcript switching.
  9. If available, test NVIDIA/CUDA diagnostics and Parakeet selection.
  10. If enabled, test Qwen cleanup on dictation, imported meeting, and recorded meeting.
"@
Set-Content -LiteralPath (Join-Path $publishDir "RELEASE-NOTES.txt") -Value $releaseNotes -Encoding UTF8

$shipChecklist = @"
Muesli Windows Ship Checklist
=============================

Build:
  [ ] dotnet build .\windows-native\Muesli.Windows\Muesli.Windows.csproj -c Release
  [ ] .\scripts\package-windows-v1.ps1     (produces ZIP + Velopack Setup.exe)
  [ ] .\scripts\test-windows-package.ps1

Installer / ZIP:
  [ ] Muesli-win-Setup.exe (Velopack) installs cleanly to %LocalAppData%\Muesli.
  [ ] ZIP extracts and smoke test launches Muesli.exe.
  [ ] About > Check Now picks up a newer published release.
  [ ] uninstall-windows.ps1 removes shortcuts/startup registration.
  [ ] install-windows.ps1 can set StartAtLogin when requested.

Core dictation:
  [ ] First-run onboarding appears only once.
  [ ] F8 works when available.
  [ ] Shortcut capture works when F8 is already in use.
  [ ] Hold shortcut records and release transcribes.
  [ ] Active-app paste works in Notepad, Chrome, Word, Outlook, Teams, Slack/Discord.
  [ ] Clipboard fallback works.

Runtime / models:
  [ ] Bundled CPython 3.12 detected (worker python resolves via 'bundled' in logs).
  [ ] Whisper dependencies OK.
  [ ] Whisper base cached and works offline.
  [ ] tiny/base/small tested on low-end CPU.
  [ ] CUDA status reported correctly on NVIDIA machine.
  [ ] Parakeet dependencies and fallback behavior tested on NVIDIA machine.
  [ ] Qwen dependencies and local model status tested.

Meetings:
  [ ] Google Meet Chrome prompt appears and does not open Outlook.
  [ ] Zoom desktop prompt appears.
  [ ] Teams prompt appears.
  [ ] Webex prompt appears.
  [ ] Dismiss, Join Only, and Join & Record all behave correctly.
  [ ] Recording creates notes/transcript and optional retained audio.

UI clone fidelity:
  [ ] Sidebar spacing and selected states match OG screenshots.
  [ ] Floating indicator idle/recording/transcribing states match OG closely.
  [ ] Meeting detection toast matches OG layout/timing.
  [ ] Dictation rows are selectable/copyable.
  [ ] Search results show dictations and meetings.
  [ ] Settings panes avoid placeholder/fake data.

Release:
  [ ] Artifact sizes recorded.
  [ ] Code signing certificate applied when available.
  [ ] Release notes reviewed.
  [ ] Known limitations documented.
"@
Set-Content -LiteralPath (Join-Path $publishDir "SHIP-CHECKLIST.txt") -Value $shipChecklist -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "setup-worker-runtime.ps1") -Destination (Join-Path $publishDir "setup-worker-runtime.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "install-windows.ps1") -Destination (Join-Path $publishDir "install-windows.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "uninstall-windows.ps1") -Destination (Join-Path $publishDir "uninstall-windows.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "test-windows-package.ps1") -Destination (Join-Path $publishDir "test-windows-package.ps1") -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "fresh-machine-qa.ps1") -Destination (Join-Path $publishDir "fresh-machine-qa.ps1") -Force
Set-Content -LiteralPath $lastPublishFile -Value $publishDir -Encoding UTF8

if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $publishDir "*") -DestinationPath $zipPath

Write-Host "Created $zipPath"

$velopackDir = Join-Path $artifactsDir "velopack"
$velopackVersion = ([xml](Get-Content (Join-Path $root "windows-native\Muesli.Windows\Muesli.Windows.csproj"))).Project.PropertyGroup.Version
if ([string]::IsNullOrWhiteSpace($velopackVersion)) {
    throw "Could not read <Version> from Muesli.Windows.csproj for Velopack packaging."
}

if (-not (Get-Command vpk -ErrorAction SilentlyContinue)) {
    Write-Host "Installing vpk CLI globally"
    dotnet tool install -g vpk
    if ($LASTEXITCODE -ne 0) { throw "dotnet tool install -g vpk failed." }
    $toolsBin = Join-Path $env:USERPROFILE ".dotnet\tools"
    if (-not ($env:PATH -split ';' | Where-Object { $_ -eq $toolsBin })) {
        $env:PATH = "$toolsBin;$env:PATH"
    }
}

New-Item -ItemType Directory -Force -Path $velopackDir | Out-Null
try {
    $prevManifest = Join-Path $velopackDir "releases.win.json"
    if (Test-Path $prevManifest) { Remove-Item -LiteralPath $prevManifest -Force }
    Invoke-WebRequest `
        -Uri "https://github.com/Muesli-HQ/Muesli-Windows/releases/latest/download/releases.win.json" `
        -OutFile $prevManifest -UseBasicParsing -ErrorAction Stop
    Write-Host "Pulled previous release manifest for delta packaging."
} catch {
    Write-Host "No previous release manifest available; full package only."
}

$icon = Join-Path $root "windows-native\Muesli.Windows\Assets\muesli.ico"
$packArgs = @(
    "pack",
    "--packId", "Muesli",
    "--packVersion", $velopackVersion,
    "--packDir", $publishDir,
    "--mainExe", "Muesli.exe",
    "--outputDir", $velopackDir,
    "--packAuthors", "Muesli",
    "--packTitle", "Muesli"
)
if (Test-Path $icon) { $packArgs += @("--icon", $icon) }
& vpk @packArgs
if ($LASTEXITCODE -ne 0) { throw "vpk pack failed with exit $LASTEXITCODE." }

Write-Host "Velopack artifacts written to $velopackDir"
