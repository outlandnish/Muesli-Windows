"""Download the Parakeet-TDT NPU model files into the model root.

The model is split across TWO HuggingFace repos (see MODEL_NOTES.md):

  * trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s  -> the Hexagon/QNN ENCODER only
        encoder-model.onnx  (408-byte EPContext wrapper)
        encoder-model.bin   (632 MB QNN context binary)

  * istupakov/parakeet-tdt-0.6b-v3-onnx     -> everything else (CPU pieces)
        nemo128.onnx                        (log-mel preprocessor)
        decoder_joint-model.int8.onnx       (decoder + joiner, FUSED)
        vocab.txt
        config.json

Downloading only the trsdn repo gets you an encoder with nothing to decode
against. Re-running is safe: huggingface_hub skips files already present.
"""
from huggingface_hub import hf_hub_download

from . import config

ENCODER_REPO = "trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s"
UPSTREAM_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx"


def fetch_all() -> None:
    """Pull every model file into the current config.model_root()."""
    enc_dir = config.encoder_dir()
    up_dir = config.upstream_dir()
    downloads = [
        # (repo_id, filename, local_dir)
        (ENCODER_REPO, "encoder-model.onnx", enc_dir),
        (ENCODER_REPO, "encoder-model.bin", enc_dir),
        (UPSTREAM_REPO, "nemo128.onnx", up_dir),
        (UPSTREAM_REPO, "decoder_joint-model.int8.onnx", up_dir),
        (UPSTREAM_REPO, "vocab.txt", up_dir),
        (UPSTREAM_REPO, "config.json", up_dir),
    ]
    config.model_root().mkdir(parents=True, exist_ok=True)
    for repo_id, filename, local_dir in downloads:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {repo_id}/{filename} -> {local_dir}/")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(local_dir))


if __name__ == "__main__":
    fetch_all()
    print("\nDone.")
