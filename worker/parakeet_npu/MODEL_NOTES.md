# Parakeet-TDT NPU engine — model notes

Verified on a Snapdragon X device (2026-07). These facts are load-bearing; if any
is wrong the engine silently emits fluent nonsense rather than erroring.

## Model files — split across TWO HuggingFace repos

`fetch_models.py` pulls both into the model root (Muesli: its `cache_dir()`).

**`trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s` — ENCODER ONLY**
- `encoder-model.onnx` — 408-byte EPContext wrapper pointing at the `.bin`
- `encoder-model.bin` — 632 MB QNN/QAIRT context binary (the Hexagon graph)

**`istupakov/parakeet-tdt-0.6b-v3-onnx` — everything else**
- `nemo128.onnx` — mel preprocessor (use it, not librosa — librosa won't
  bit-match NeMo → fluent garbage). onnx-asr bundles its own copy.
- `decoder_joint-model.int8.onnx` — decoder + joiner FUSED into one graph
- `vocab.txt`; `config.json` (`features_size:128, subsampling_factor:8`)

## Shapes / vocab

- **Mel bins 128.** Encoder in: `audio_signal` f32 `[1,128,801]` + `length` i32
  `[1]`; out: `output_0` f32 `[1,1024,101]`. Fixed 8 s window: 801 mel → 101 enc
  frames. `1024` is the hidden dim, not the vocab.
- **Vocab 8193 tokens** (`<blk>` at 8192, inside the 8193).
- **TDT joint width 8198** = 8193 vocab + 5 durations {0,1,2,3,4}. onnx-asr owns
  the split and the decode loop.

## Why the engine is structured the way it is

- **Subclass onnx-asr, override only the encoder.** onnx-asr's `NemoConformerTdt`
  assumes a dynamic-length CPU encoder over the whole utterance, incompatible with
  the fixed 8 s Hexagon encoder. We reuse its preprocessor/vocab/decoder_joint/TDT
  decode and swap in the QNN encoder session. Reimplementing the TDT decode by hand
  is the correctness-critical code we deliberately avoid.
- **Decode each 8 s window independently, then merge the texts.** Stitching encoder
  *frames* and decoding once corrupts the decoder's LSTM state at the seams and
  truncates the transcript. Windows overlap ~2 s; `_merge` fuzzy-matches the word
  overlap and splices at its midpoint.

## Runtime requirements (hard)

- **Native ARM64 Python.** `platform.machine()` reports the host CPU (ARM64) even
  for an x64-emulated interpreter — not a reliable check. An x64 interpreter cannot
  load the native-ARM64 QNN DLLs (`Error 193`).
- **onnxruntime-qnn 2.x is a PLUGIN EP**, not a replacement onnxruntime:
  `register_execution_provider_library(...)` then select the NPU by hardware type
  (`OrtHardwareDeviceType.NPU`). `asr.qnn_npu_available()` probes this.

## Performance

Parakeet encode is compute-bound (Conformer over an 8 s window, weights reused
across frames) → the Hexagon matrix engine stays fed → ~20x realtime on-device.
Validated against a CPU reference transcript; silence → empty.
