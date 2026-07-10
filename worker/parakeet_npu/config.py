"""Config for the Parakeet-TDT NPU engine (encoder on the Hexagon NPU).

Vendored from local-npu-notes and adapted for Muesli: model files resolve under
a caller-supplied base directory (Muesli passes its `cache_dir()`), not a
repo-local `models/`. Set the base once via `set_model_root(...)` before
constructing the ASR, or via the `MUESLI_PARAKEET_NPU_MODEL_DIR` env var.

The values marked (VERIFY) must match the compiled model — they are the usual
source of silent breakage. See MODEL_NOTES.md.
"""
import os
from pathlib import Path

# --- Audio ------------------------------------------------------------------
TARGET_SAMPLE_RATE = 16000      # Parakeet expects 16 kHz mono
CHANNELS = 1

# --- Fixed encoder window ---------------------------------------------------
# The Hexagon encoder is compiled for a fixed 8-second window.
WINDOW_SECONDS = 8.0
OVERLAP_SECONDS = 1.0

# --- Log-mel featurizer -----------------------------------------------------
# Mel featurizing is done by the model's own preprocessor graph (nemo128.onnx),
# NOT hand-rolled — a librosa mel will never bit-match NeMo and produces
# fluent-garbage transcripts. N_MELS/MEL_FRAMES are asserted against reality.
N_MELS = 128                    # CONFIRMED: features_size=128 upstream
MEL_FRAMES = 801                # 8 s @ 100 fps (+1); encoder input is [1,128,801]
MEL_SUBSAMPLE = 8               # encoder subsampling: 801 mel -> 101 enc frames

# --- Model files (split across TWO HuggingFace repos) -----------------------
# See MODEL_NOTES.md. The trsdn "-8s" repo is ENCODER-ONLY; everything else
# (mel preprocessor, fused decoder+joiner, vocab) comes from the istupakov repo.
# fetch_models.py pulls both into the model root below.
_ENV_MODEL_DIR = "MUESLI_PARAKEET_NPU_MODEL_DIR"


def model_root() -> Path:
    """Base directory holding the two model repos. Muesli sets the env var to
    its own model cache; falls back to a repo-local ./models for standalone use."""
    root = os.environ.get(_ENV_MODEL_DIR)
    return Path(root) if root else Path("models")


def set_model_root(path) -> None:
    os.environ[_ENV_MODEL_DIR] = str(path)


def encoder_dir() -> Path:
    return model_root() / "parakeet-tdt-0.6b-v3-htp-int8-8s"


def upstream_dir() -> Path:
    return model_root() / "parakeet-tdt-0.6b-v3-onnx"


# From trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s (Hexagon/QNN encoder):
def encoder_qnn() -> Path:
    return encoder_dir() / "encoder-model.onnx"   # 408-byte EPContext wrapper -> .bin


# From istupakov/parakeet-tdt-0.6b-v3-onnx (dynamic-shape CPU pieces):
def preprocessor_onnx() -> Path:
    return upstream_dir() / "nemo128.onnx"


def decoder_joint_onnx() -> Path:
    return upstream_dir() / "decoder_joint-model.int8.onnx"


def vocab_txt() -> Path:
    return upstream_dir() / "vocab.txt"


def upstream_config_json() -> Path:
    return upstream_dir() / "config.json"


# TDT specifics. MEASURED on-device: joint output last dim = 8198 = 8193 vocab
# (<blk> is index 8192, already inside the vocab) + 5 durations {0,1,2,3,4}.
TDT_DURATIONS = [0, 1, 2, 3, 4]


def models_present() -> bool:
    """True iff every model file is on disk under the current model_root()."""
    return all(
        p().exists()
        for p in (encoder_qnn, preprocessor_onnx, decoder_joint_onnx,
                  vocab_txt, upstream_config_json)
    )


def validate():
    """Assert the invariants that, if wrong, silently produce garbage transcripts
    rather than an error. Called at ASR construction."""
    assert N_MELS == 128, f"encoder expects 128 mel bins, got {N_MELS}"
    assert MEL_FRAMES == 801, f"encoder window is 801 mel frames, got {MEL_FRAMES}"
    assert MEL_SUBSAMPLE == 8, f"encoder subsampling is 8x, got {MEL_SUBSAMPLE}"
    assert TARGET_SAMPLE_RATE == 16000, "Parakeet expects 16 kHz"
    assert 0 < OVERLAP_SECONDS < WINDOW_SECONDS, "overlap must be within the window"
