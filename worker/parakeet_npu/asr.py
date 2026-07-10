"""Parakeet TDT ASR with the encoder on the Hexagon NPU.

Vendored from local-npu-notes for Muesli. Reuses onnx-asr's nemo128 preprocessor,
vocab, fused decoder_joint and TDT decode; overrides only the encoder to run the
compiled Hexagon build. The encoder is fixed at an 801-mel-frame (8 s) window, so
long audio is processed as overlapping windows that are each decoded independently
and merged at the text level (decoding a stitched feature stream corrupts the
decoder state at the seams).

Requires a native ARM64 Python + onnxruntime-qnn. See MODEL_NOTES.md.
"""
import numpy as np
import onnxruntime as ort

from onnx_asr.asr import _AsrWithDecoding
from onnx_asr.models.nemo import NemoConformerTdt
from onnx_asr.preprocessors.numpy_preprocessor import NemoPreprocessorNumpy
from onnx_asr.preprocessors.preprocessor import OnnxPreprocessor

from . import config

ENC_IN_FRAMES = config.MEL_FRAMES          # 801, fixed by the .bin
WINDOW_MEL = 800                           # usable mel frames per window
OVERLAP_MEL = 200                          # ~2 s overlap between windows
HOP_MEL = WINDOW_MEL - OVERLAP_MEL          # 600
# Seconds per encoder frame: mel hop 160 @ 16 kHz, subsampled by MEL_SUBSAMPLE (8).
# = 8 * 160 / 16000 = 0.08 s — a token's encoder-frame index * this = its time.
SEC_PER_ENC_FRAME = config.MEL_SUBSAMPLE * 160 / config.TARGET_SAMPLE_RATE

_QNN_EP = "QNNExecutionProvider"
_qnn_registered = False
_qnn_htp_path = None


def _ensure_qnn_registered():
    # onnxruntime-qnn 2.x is a plugin EP: register its library, then select a
    # device (no providers=[...] path). Requires a native ARM64 Python.
    global _qnn_registered, _qnn_htp_path
    if _qnn_registered:
        return
    try:
        import onnxruntime_qnn as _qnn
    except ImportError as e:
        raise RuntimeError(
            "onnxruntime-qnn not installed. Use a native ARM64 Python and "
            "`pip install onnxruntime-qnn` (no --no-deps)."
        ) from e
    ort.register_execution_provider_library(_QNN_EP, _qnn.get_library_path())
    _qnn_htp_path = _qnn.get_qnn_htp_path()
    _qnn_registered = True


def qnn_npu_available() -> bool:
    """True iff the QNN plugin loads and exposes an NPU device. Cheap probe used
    by the worker to gate the engine (mirrors has_nvidia_runtime for CUDA)."""
    try:
        _ensure_qnn_registered()
        npu = ort.OrtHardwareDeviceType.NPU
        return any(
            d.ep_name == _QNN_EP and d.device.type == npu
            for d in ort.get_ep_devices()
        )
    except Exception:
        return False


def _qnn_htp_device():
    npu = ort.OrtHardwareDeviceType.NPU
    for d in ort.get_ep_devices():
        if d.ep_name == _QNN_EP and d.device.type == npu:
            return d
    raise RuntimeError(
        "No QNN NPU device found (seen: "
        + repr([str(d.device.type) for d in ort.get_ep_devices()
                if d.ep_name == _QNN_EP]) + ")."
    )


def _qnn_soc_description():
    """Best-effort SoC model string from QNN device metadata, e.g.
    'Snapdragon(R) X 12-core X1E80100 @ 3.40 GHz'. The CPU device carries the
    cleanest name; fall back to any QNN device's Description. Returns '' if none."""
    try:
        _ensure_qnn_registered()
    except Exception:
        return ""
    best = ""
    for d in ort.get_ep_devices():
        if d.ep_name != _QNN_EP:
            continue
        md = getattr(d, "metadata", None) or getattr(d.device, "metadata", None)
        desc = ""
        try:
            desc = md["Description"] if md and "Description" in md else ""
        except Exception:
            desc = ""
        if desc and "Snapdragon" in desc:
            return desc            # a Snapdragon SoC string is the best answer
        best = best or desc
    return best


# The Hexagon encoder is a pre-compiled QNN context binary tied to an HTP
# architecture. Support is therefore per-SoC, and only what we've actually run is
# "verified". Everything else is a best-effort attempt that fails safe: if the
# context binary won't load on a given SoC, session construction raises and the
# worker degrades to its mock fallback rather than crashing.
_VERIFIED_SOCS = {"X1E80100"}       # Snapdragon X Elite — validated on-device


def qnn_soc_support():
    """Classify the current SoC for the parakeet NPU engine. Returns a dict:
        soc:   the raw SoC model number found (e.g. 'X1E80100'), or ''
        desc:  the full SoC description string, or ''
        tier:  'verified' | 'elite-untested' | 'plus-untested' | 'unknown'
        note:  human-readable support note for warnings/UI
    This never blocks the engine — it only informs. The real gate is whether the
    context binary loads (attempted at session construction)."""
    import re
    desc = _qnn_soc_description()
    m = re.search(r"\bX1[EP]\d{5}\b", desc or "", re.IGNORECASE)
    soc = m.group(0).upper() if m else ""
    if soc in _VERIFIED_SOCS:
        tier = "verified"
        note = f"{soc}: validated on-device."
    elif soc.startswith("X1E"):
        tier = "elite-untested"
        note = (f"{soc}: Snapdragon X Elite, same NPU generation as the validated "
                "part but not individually tested — expected to work.")
    elif soc.startswith("X1P"):
        tier = "plus-untested"
        note = (f"{soc}: Snapdragon X Plus — not yet validated; the encoder may need "
                "a Plus-targeted build. Will attempt and fall back if it can't load.")
    else:
        tier = "unknown"
        note = (f"Unrecognized Snapdragon SoC ({desc or 'unknown'}). Will attempt the "
                "NPU path and fall back safely if the encoder can't load.")
    return {"soc": soc, "desc": desc, "tier": tier, "note": note}


def _qnn_encoder_session(path):
    _ensure_qnn_registered()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.add_provider_for_devices([_qnn_htp_device()], {"backend_path": _qnn_htp_path})
    return ort.InferenceSession(str(path), sess_options=so)


def _merge(prev_words, nxt, max_overlap=30):
    """Append nxt's words to prev_words, dropping the overlapping span. Adjacent
    chunks share ~overlap seconds of audio, transcribed slightly differently at
    the seam, so we find the overlap length k where prev's last k words best
    match nxt's first k, and splice in the middle of it (so neither side's
    divergent boundary word strands)."""
    a, b = prev_words, nxt.split()
    if not a or not b:
        return a + b

    def norm(w):
        return w.lower().strip(".,!?\"'")

    na, nb = [norm(w) for w in a], [norm(w) for w in b]
    span = min(len(a), len(b), max_overlap)
    best_k, best_score = 0, 0.0
    for k in range(2, span + 1):
        score = sum(na[len(a) - k + t] == nb[t] for t in range(k)) / k
        if score >= 0.6 and score >= best_score:
            best_score, best_k = score, k
    if best_k:
        cut = best_k // 2
        return a[: len(a) - best_k + cut] + b[cut:]
    # No multi-word overlap. Still drop a single duplicated boundary word (the
    # common "working. / working," stutter at a seam).
    if na[-1] == nb[0]:
        return a + b[1:]
    return a + b


class HexagonParakeet(NemoConformerTdt):
    def __init__(self, model_files, preprocessor_factory, onnx_options):
        # Bypass NemoConformerRnnt.__init__ (builds a CPU encoder with the wrong
        # I/O); init the base for vocab/preprocessor, then build our own sessions.
        _AsrWithDecoding.__init__(self, model_files, preprocessor_factory, onnx_options)
        self._encoder = _qnn_encoder_session(model_files["encoder"])
        self._decoder_joint = ort.InferenceSession(
            str(model_files["decoder_joint"]), **onnx_options
        )
        self._enc_in = [i.name for i in self._encoder.get_inputs()]
        self._enc_out = [o.name for o in self._encoder.get_outputs()]

    def _run_window(self, feats_win):
        # feats_win (1,128,801) -> encoder output_0 (1,1024,101)
        length = np.array([feats_win.shape[-1]], dtype=np.int32)
        return self._encoder.run(
            self._enc_out,
            {self._enc_in[0]: feats_win, self._enc_in[1]: length},
        )[0]

    def _decode_window(self, feats_win, valid_mel):
        enc = self._run_window(feats_win)                        # (1,1024,101)
        valid_enc = -(-valid_mel // config.MEL_SUBSAMPLE)        # ceil(mel/8)
        enc = enc[:, :, :valid_enc].transpose(0, 2, 1)           # (1,T,1024)
        lens = np.array([enc.shape[1]], dtype=np.int64)
        results = list(self._decoding(enc.astype(np.float32), lens))
        if not results:
            return ""
        return "".join(self._vocab[i] for i in results[0][0]).replace("▁", " ").strip()

    def _decode_window_words(self, feats_win, valid_mel, win_offset_s):
        """Like _decode_window but returns [(word, start_s, end_s), ...] with times
        absolute in the utterance. Uses the transducer's per-token encoder-frame
        timestamps (results[0][1]); a word's span runs from its first sub-token's
        time to the next word's start."""
        enc = self._run_window(feats_win)
        valid_enc = -(-valid_mel // config.MEL_SUBSAMPLE)
        enc = enc[:, :, :valid_enc].transpose(0, 2, 1)
        lens = np.array([enc.shape[1]], dtype=np.int64)
        results = list(self._decoding(enc.astype(np.float32), lens))
        if not results:
            return []
        tokens, timestamps = results[0][0], results[0][1]
        # Group sub-word tokens into words. This tokenizer marks a word start with a
        # LEADING SPACE (not "▁"); skip special <|...|> control tokens.
        words = []
        cur, cur_start = "", None
        for tok, ts in zip(tokens, timestamps):
            piece = self._vocab[tok]
            if piece.startswith("<") and piece.endswith(">"):
                continue  # control token, no text/time
            t_s = win_offset_s + ts * SEC_PER_ENC_FRAME
            if piece.startswith(" "):
                if cur.strip():
                    words.append([cur.strip(), cur_start, t_s])
                cur, cur_start = piece, t_s
            else:
                if cur_start is None:
                    cur_start = t_s
                cur += piece
        if cur.strip():
            words.append([cur.strip(), cur_start, None])
        return words

    def recognize_batch(self, waveforms, waveforms_len, /, **kwargs):
        # onnx-asr's base recognize_batch does one _encode -> one _decoding, which
        # our fixed-window encoder can't do. Instead: window the mel features,
        # decode each window independently, and merge the texts.
        from onnx_asr.asr import TimestampedResult
        out = []
        for wav, wlen in zip(waveforms, waveforms_len):
            feats, _ = self._preprocessor(wav[None], np.array([wlen], np.int64))
            feats = feats.astype(np.float32)
            _, n_mels, total = feats.shape
            words, start = [], 0
            while start < total:
                win = feats[:, :, start:start + WINDOW_MEL]
                valid = win.shape[-1]
                buf = np.zeros((1, n_mels, ENC_IN_FRAMES), dtype=np.float32)
                buf[:, :, :valid] = win
                text = self._decode_window(buf, valid)
                words = _merge(words, text) if words else text.split()
                if valid < WINDOW_MEL:
                    break
                start += HOP_MEL
            out.append(TimestampedResult(" ".join(words), None, None, None))
        return iter(out)

    def recognize_words(self, wav, wlen):
        """Return [(word, start_s, end_s), ...] for one waveform, timed absolute in
        the utterance. Windows like recognize_batch, but keeps per-word times and
        drops words falling in a window's leading overlap (already emitted by the
        previous window) so each word appears once."""
        feats, _ = self._preprocessor(wav[None], np.array([wlen], np.int64))
        feats = feats.astype(np.float32)
        _, n_mels, total = feats.shape
        all_words, start = [], 0
        while start < total:
            win = feats[:, :, start:start + WINDOW_MEL]
            valid = win.shape[-1]
            buf = np.zeros((1, n_mels, ENC_IN_FRAMES), dtype=np.float32)
            buf[:, :, :valid] = win
            # `start` is a MEL-frame offset; mel hop = 160 samples @ 16 kHz = 0.01 s.
            win_offset_s = start * 160 / config.TARGET_SAMPLE_RATE
            words = self._decode_window_words(buf, valid, win_offset_s)
            # Drop the leading-overlap words this window re-decoded from the previous
            # one. The two windows transcribe the shared ~2 s overlap slightly
            # differently, so we can't rely on exact text; combine two temporal rules:
            #  (a) drop any new word starting at/before the last kept word (re-decode
            #      of already-covered audio, and it would scramble the time order);
            #  (b) within the overlap region, also drop a new word that repeats a
            #      just-kept word (same normalized text within ~0.6 s) — this catches
            #      re-decodes that land a hair later, WITHOUT touching genuine
            #      disfluencies (which come from a single window, already merged, and
            #      sit outside the fresh window's leading-overlap span).
            if start > 0 and all_words and words:
                covered_until = all_words[-1][1] - 0.05
                overlap_end_s = win_offset_s + (OVERLAP_MEL * 160 / config.TARGET_SAMPLE_RATE)
                kept = []
                for w in words:
                    if w[1] <= covered_until:
                        continue
                    if w[1] < overlap_end_s and all_words:
                        pw = all_words[-1] if not kept else kept[-1]
                        same = w[0].lower().strip(".,!?\"'") == pw[0].lower().strip(".,!?\"'")
                        if same and abs(w[1] - pw[1]) < 0.6:
                            continue
                    kept.append(w)
                words = kept
            all_words.extend(words)
            if valid < WINDOW_MEL:
                break
            start += HOP_MEL
        # fill trailing end times (each word ends where the next begins)
        for i in range(len(all_words) - 1):
            if all_words[i][2] is None:
                all_words[i][2] = all_words[i + 1][1]
        if all_words and all_words[-1][2] is None:
            all_words[-1][2] = all_words[-1][1] + 0.4
        return [(w, s, e) for w, s, e in all_words]


def _make_preprocessor(name):
    opts = {"sess_options": None, "providers": None, "provider_options": None}
    try:
        return OnnxPreprocessor(name, opts)
    except Exception:
        return NemoPreprocessorNumpy(name)


class ParakeetTDT:
    """Wrapper over HexagonParakeet with a plain transcribe(wav) method."""

    def __init__(self):
        config.validate()
        model_files = {
            "encoder": config.encoder_qnn(),
            "decoder_joint": config.decoder_joint_onnx(),
            "vocab": config.vocab_txt(),
            "config": config.upstream_config_json(),
        }
        onnx_options = {"sess_options": None, "providers": None, "provider_options": None}
        self.model = HexagonParakeet(model_files, _make_preprocessor, onnx_options)

    def transcribe(self, wav16k: np.ndarray) -> str:
        wav = np.ascontiguousarray(wav16k, dtype=np.float32)[None]
        lens = np.array([wav.shape[1]], dtype=np.int64)
        results = list(self.model.recognize_batch(wav, lens))
        return results[0].text if results else ""

    def transcribe_words(self, wav16k: np.ndarray):
        """Return [(word, start_s, end_s), ...] with per-word timestamps, for
        speaker-attributed transcripts. Same decode as transcribe(), plus timing."""
        wav = np.ascontiguousarray(wav16k, dtype=np.float32)[None]
        return self.model.recognize_words(wav[0], wav.shape[1])
