using System.Diagnostics;
using System.IO;
using System.Net.Http;

namespace Muesli.Windows.Services;

/// <summary>
/// Manages the GenieX on-device LLM server (BSD-3-Clause; https://github.com/qualcomm/GenieX)
/// used by the "npu" meeting-summary provider. GenieX exposes an OpenAI-compatible
/// API at http://127.0.0.1:18181/v1.
///
/// Responsibilities, mirroring how TranscriptionWorkerClient manages the Python
/// worker: locate geniex.exe, ensure the summary model is pulled, and start /
/// reuse / stop the `geniex serve` process. If GenieX isn't installed we surface
/// that clearly rather than failing opaquely — it's an optional add-on
/// (`setup-worker-runtime.ps1 -WithNpuSummary`, which pip-installs geniex into the
/// bundled runtime), not a base dependency.
/// </summary>
public sealed class GenieXService : IDisposable
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromSeconds(5) };
    private readonly object _gate = new();
    private Process? _serverProcess;   // non-null only if WE started it
    private bool _weStartedIt;

    public string BaseUrl { get; }
    private readonly string _healthUrl;

    public GenieXService(string baseUrl = "http://127.0.0.1:18181/v1")
    {
        BaseUrl = baseUrl.TrimEnd('/');
        _healthUrl = $"{BaseUrl}/models";
    }

    /// <summary>Locate geniex.exe: explicit override, the bundled runtime's
    /// Scripts dir (pip install geniex), PATH, then the standalone-installer
    /// location. Returns null if not found.</summary>
    public static string? FindGenieXExecutable()
    {
        var overridePath = Environment.GetEnvironmentVariable("MUESLI_GENIEX_PATH");
        if (!string.IsNullOrWhiteSpace(overridePath) && File.Exists(overridePath))
        {
            return overridePath;
        }

        // Bundled runtime: `pip install geniex` drops geniex.exe next to python.
        try
        {
            var python = WorkerRuntimeLocator.FindPythonExecutable();
            var pyDir = Path.GetDirectoryName(python);
            if (!string.IsNullOrEmpty(pyDir))
            {
                foreach (var candidate in new[]
                {
                    Path.Combine(pyDir, "geniex.exe"),
                    Path.Combine(pyDir, "Scripts", "geniex.exe"),
                })
                {
                    if (File.Exists(candidate))
                    {
                        return candidate;
                    }
                }
            }
        }
        catch
        {
            // fall through to PATH / standalone locations
        }

        var onPath = FindOnPath("geniex.exe");
        if (onPath != null)
        {
            return onPath;
        }

        var standalone = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "GenieX CLI", "geniex.exe");
        return File.Exists(standalone) ? standalone : null;
    }

    public static bool IsInstalled() => FindGenieXExecutable() != null;

    /// <summary>True if something already answers on the GenieX port.</summary>
    public async Task<bool> IsServingAsync(CancellationToken ct = default)
    {
        try
        {
            using var response = await Http.GetAsync(_healthUrl, ct);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Ensure a GenieX server is reachable, starting one if needed.
    /// Reuses an already-running server (e.g. a dev instance) rather than
    /// spawning a second. Ensures the model is pulled first. Throws with a clear
    /// message if GenieX isn't installed.</summary>
    public async Task EnsureServingAsync(string model, Action<string>? onProgress = null, CancellationToken ct = default)
    {
        if (await IsServingAsync(ct))
        {
            return;
        }

        var geniex = FindGenieXExecutable()
            ?? throw new InvalidOperationException(
                "GenieX is not installed. Add it with the optional NPU summary setup "
                + "(setup-worker-runtime.ps1 -WithNpuSummary), or install it from "
                + "https://github.com/qualcomm/GenieX.");

        onProgress?.Invoke("Ensuring the summary model is available...");
        await EnsureModelPulledAsync(geniex, model, onProgress, ct);

        onProgress?.Invoke("Starting the on-device LLM server...");
        StartServer(geniex);

        // Wait for the server to answer (first boot loads the model — allow time).
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(120);
        while (DateTime.UtcNow < deadline)
        {
            ct.ThrowIfCancellationRequested();
            if (await IsServingAsync(ct))
            {
                onProgress?.Invoke("On-device LLM server is ready.");
                return;
            }
            await Task.Delay(1000, ct);
        }

        throw new TimeoutException("GenieX server did not become ready within 120s.");
    }

    private async Task EnsureModelPulledAsync(string geniex, string model, Action<string>? onProgress, CancellationToken ct)
    {
        // `geniex list` shows cached models; only pull if absent (a pull is a large
        // download and slow, so we avoid it when the model is already present).
        // The chat/serve model id carries a quant suffix (":Q4_0", ":W4A16") that
        // `geniex list` prints in a separate column, so compare on the base id.
        if (await IsModelCachedAsync(geniex, model, ct))
        {
            return;
        }

        onProgress?.Invoke($"Downloading {model} (first run only)...");
        var pull = await RunCaptureAsync(geniex, $"pull {model}", TimeSpan.FromMinutes(30), ct);
        if (!await IsModelCachedAsync(geniex, model, ct))
        {
            throw new InvalidOperationException($"Failed to pull GenieX model '{model}'. Output:\n{pull}");
        }
    }

    private async Task<bool> IsModelCachedAsync(string geniex, string model, CancellationToken ct)
    {
        var listed = await RunCaptureAsync(geniex, "list", TimeSpan.FromSeconds(20), ct);
        var baseId = model.Split(':')[0];   // drop the ":Q4_0" / ":W4A16" quant suffix
        return listed.Contains(baseId, StringComparison.OrdinalIgnoreCase);
    }

    private void StartServer(string geniex)
    {
        lock (_gate)
        {
            if (_serverProcess is { HasExited: false })
            {
                return;
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = geniex,
                // Long keepalive: GenieX unloads on its idle timer, and a request
                // arriving mid-unload can crash the server — keep it resident.
                Arguments = "serve --keepalive 100000",
                UseShellExecute = false,
                CreateNoWindow = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            _serverProcess = Process.Start(startInfo);
            _weStartedIt = _serverProcess != null;
        }
    }

    private static async Task<string> RunCaptureAsync(string exe, string args, TimeSpan timeout, CancellationToken ct)
    {
        var startInfo = new ProcessStartInfo
        {
            FileName = exe,
            Arguments = args,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        using var process = new Process { StartInfo = startInfo };
        process.Start();
        var stdoutTask = process.StandardOutput.ReadToEndAsync(ct);
        var stderrTask = process.StandardError.ReadToEndAsync(ct);
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(timeout);
        try
        {
            await process.WaitForExitAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            try { process.Kill(true); } catch { /* best effort */ }
        }
        return (await stdoutTask) + "\n" + (await stderrTask);
    }

    private static string? FindOnPath(string fileName)
    {
        var path = Environment.GetEnvironmentVariable("PATH") ?? "";
        foreach (var dir in path.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            try
            {
                var candidate = Path.Combine(dir.Trim(), fileName);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
            catch
            {
                // ignore malformed PATH entries
            }
        }
        return null;
    }

    public void Dispose()
    {
        // Only stop the server if we started it — never kill a user's own instance.
        lock (_gate)
        {
            if (_weStartedIt && _serverProcess is { HasExited: false })
            {
                try { _serverProcess.Kill(true); } catch { /* best effort */ }
            }
            _serverProcess?.Dispose();
            _serverProcess = null;
        }
    }
}
