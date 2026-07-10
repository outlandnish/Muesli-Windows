"""Sherpa-onnx CPU speaker diarization backend for the Muesli worker.

Used on the arm64 / Snapdragon build, where pyannote.audio's torch backend has
no win_arm64 wheel. transcribe_worker.diarize_audio() falls back to this backend
when pyannote.audio is unavailable.

Models live under cache_dir()/diarize-sherpa/ (fetched by
download_model(kind="diarize-sherpa")):
  - segmentation.onnx : sherpa-onnx-pyannote-segmentation-3-0 (must be the
                        sherpa-packaged export; the raw onnx-community export
                        lacks the 'sample_rate' metadata sherpa requires)
  - embedding.onnx    : 3dspeaker eres2net_base_sv 16k

Emits Muesli's standard segment schema: {id, speaker, startMs, endMs, text},
with anonymous, 1-indexed "Speaker N" labels.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

SHERPA_DIAR_SUBDIR = "diarize-sherpa"

_SHERPA_CACHE: dict[str, Any] = {}


def sherpa_model_dir(cache_dir: Path) -> Path:
    d = cache_dir / SHERPA_DIAR_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def sherpa_models_present(cache_dir: Path) -> bool:
    d = sherpa_model_dir(cache_dir)
    return (d / "segmentation.onnx").is_file() and (d / "embedding.onnx").is_file()


def sherpa_available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return True


def _resample_linear(audio, sr_in: int, sr_out: int):
    import numpy as np

    if sr_in == sr_out:
        return audio
    n_out = int(math.floor(len(audio) * sr_out / sr_in))
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def get_diarizer(cache_dir: Path, num_speakers: int = -1, cluster_threshold: float = 0.5):
    """Construct and cache the OfflineSpeakerDiarization. num_speakers=-1 => auto.

    Uses sherpa's default cluster_threshold (0.5). Note: unsupervised speaker-count
    detection from mixed audio is inherently approximate and content-dependent — it
    over-splits on some material and merges similar voices on others; no single
    threshold is right for all recordings. This CPU diarizer is a best-effort
    FALLBACK for attribution; for real meetings the intended source of speaker
    identity is the meeting platform (see MEETING_SPEAKERS_PLAN.md). If a recording
    has a known speaker count, pass num_speakers; the threshold can be tuned per
    run via MUESLI_DIARIZE_THRESHOLD (higher => fewer speakers).
    """
    import os

    env_threshold = os.environ.get("MUESLI_DIARIZE_THRESHOLD")
    if env_threshold:
        try:
            cluster_threshold = float(env_threshold)
        except ValueError:
            pass
    if "diarizer" in _SHERPA_CACHE:
        return _SHERPA_CACHE["diarizer"]

    import sherpa_onnx

    d = sherpa_model_dir(cache_dir)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(d / "segmentation.onnx")
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(d / "embedding.onnx")
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("sherpa OfflineSpeakerDiarizationConfig.validate() failed")

    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    _SHERPA_CACHE["diarizer"] = diarizer
    return diarizer


def diarize_audio_sherpa(input_path: str, cache_dir: Path) -> dict[str, Any]:
    """Diarize with sherpa-onnx on the CPU. Same return schema as diarize_audio()."""
    import soundfile as sf

    warnings: list[str] = []
    started_at = time.perf_counter()
    try:
        if not sherpa_models_present(cache_dir):
            warnings += [
                "Diarization skipped: sherpa diarization models not downloaded.",
                "Fetch them first via download_model(kind='diarize-sherpa').",
            ]
            return _empty(warnings)

        diarizer = get_diarizer(cache_dir)
        model_ms = (time.perf_counter() - started_at) * 1000

        audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
        audio = audio[:, 0]  # first channel only
        audio = _resample_linear(audio, sr, diarizer.sample_rate)

        infer_started_at = time.perf_counter()
        result = diarizer.process(audio).sort_by_start_time()
        infer_ms = (time.perf_counter() - infer_started_at) * 1000

        # sherpa's raw cluster ids are arbitrary (may not start at 0 or be
        # contiguous). Remap to a dense 1-based sequence in order of first
        # appearance so labels read "Speaker 1", "Speaker 2", ...
        label_of: dict[int, str] = {}
        segments = []
        for r in result:
            if r.speaker not in label_of:
                label_of[r.speaker] = f"Speaker {len(label_of) + 1}"
            segments.append(
                {
                    "id": f"diarize_{len(segments)}",
                    "speaker": label_of[r.speaker],
                    "startMs": int(r.start * 1000),
                    "endMs": int(r.end * 1000),
                    "text": "",
                }
            )

        total_ms = (time.perf_counter() - started_at) * 1000
        warnings += [
            "ASR engine: diarization",
            "Diarization backend: sherpa-onnx / CPU (ARM64)",
            f"Diarization segments: {len(segments)}",
            f"Timing diarization model ms: {model_ms:.1f}",
            f"Timing diarization infer ms: {infer_ms:.1f}",
            f"Timing diarization total ms: {total_ms:.1f}",
        ]
        return {
            "transcriptText": f"Diarized {len(segments)} segments.",
            "detectedLanguage": "en",
            "durationMs": 0,
            "segments": segments,
            "warnings": warnings,
        }
    except Exception as exc:  # noqa: BLE001
        warnings += ["Diarization failed (sherpa).", f"Error: {exc}"]
        return _empty(warnings)


def _empty(warnings: list[str]) -> dict[str, Any]:
    return {
        "transcriptText": "",
        "detectedLanguage": "en",
        "durationMs": 0,
        "segments": [],
        "warnings": warnings,
    }
