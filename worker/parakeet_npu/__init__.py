"""Parakeet-TDT ASR with the encoder on the Qualcomm Hexagon NPU (QNN).

Vendored from local-npu-notes. Public surface used by transcribe_worker.py:

    from parakeet_npu import config, fetch_models
    from parakeet_npu.asr import ParakeetTDT, qnn_npu_available

Requires a native ARM64 Python + onnxruntime-qnn (Snapdragon X). See MODEL_NOTES.md.
"""
