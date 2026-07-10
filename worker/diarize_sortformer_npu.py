"""NVIDIA Sortformer streaming speaker diarization on the Qualcomm Hexagon NPU.

End-to-end 4-speaker diarizer (VAD + speaker-change + overlap + attribution in ONE
model) — replaces pyannote-seg + wespeaker-embed + clustering. Runs the compiled
QNN context binary via onnxruntime-qnn, the same plugin-EP pattern as the
parakeet-v3-npu ASR encoder. Falls back to the float ONNX on CPU when no NPU is
available (both are self-consistent; the loop auto-detects static vs dynamic graph).

transcribe_worker.diarize_audio() prefers this backend on arm64/Snapdragon when the
NPU is present, ahead of the sherpa-onnx CPU backend.

Models live under cache_dir()/diarize-sortformer/ (fetched by
download_model(kind="diarize-sortformer-npu")):
  - sortformer.bin  : compiled QNN context binary (NPU), OR
  - sortformer.onnx : float ONNX (CPU fallback)

Model provenance: cgus/diar_streaming_sortformer_4spk-v2.1-onnx (ONNX export of
nvidia/diar_streaming_sortformer_4spk-v2.1). Streaming loop + featurizer + post-
processing ported from altunenes/parakeet-rs src/sortformer.rs.

Emits Muesli's standard segment schema {id, speaker, startMs, endMs, text} with
anonymous, 1-indexed "Speaker N" labels.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

SORTFORMER_SUBDIR = "diarize-sortformer"

# --- feature constants (Sortformer; normalize='NA' = NO per-utterance mel norm) ---
N_FFT = 512; WIN = 400; HOP = 160; N_MELS = 128; PREEMPH = 0.97
LOG_ZERO_GUARD = float(2.0 ** -24); SR = 16000
# --- streaming constants (parakeet-rs defaults; ONNX carries no metadata) ---
CHUNK_LEN = 124; FIFO_LEN = 124; SPKCACHE_LEN = 188; RC = 1; SUB = 8
EMB = 512; NSPK = 4; FRAME_DUR = 0.08
FEED = (CHUNK_LEN + RC) * SUB          # 1000 mel frames
CHUNK_STRIDE = CHUNK_LEN * SUB         # 992

# Static/frozen graph (NPU) warm-up: the frozen cache shapes make the model
# under-detect until the caches fill (cold-start). Re-processing a ~10 s pre-roll of
# the clip's own start warms the caches, then those frames are discarded. Recovers
# ~70% of the static-vs-dynamic accuracy gap (measured vs Silero VAD on a real
# meeting-length clip: 88.4% -> 92.8% F1, vs 94.6% dynamic). Only the static/NPU
# path needs this; the dynamic ONNX handles the true cold-start correctly.
WARMUP_SEC = 10.0
WARMUP_FRAMES = int(WARMUP_SEC / FRAME_DUR)   # 125 output frames

# --- NeMo speaker-cache smart-compression params (parakeet-rs sortformer.rs) ---
SPKCACHE_SIL_FRAMES_PER_SPK = 3
PRED_SCORE_THRESHOLD = 0.25
STRONG_BOOST_RATE = 0.75
WEAK_BOOST_RATE = 1.5
MIN_POS_SCORES_RATE = 0.5
SIL_THRESHOLD = 0.2
MAX_INDEX = 99999

# CallHome post-processing config (parakeet-rs DiarizationConfig::callhome)
CFG = dict(onset=0.641, offset=0.561, pad_onset=0.229, pad_offset=0.079,
           min_duration_on=0.511, min_duration_off=0.296, median_window=11)

_CACHE: dict[str, Any] = {}
_QNN_EP = "QNNExecutionProvider"
_qnn_registered = False
_qnn_htp_path = None


# ---------- model dir / presence ----------
def sortformer_model_dir(cache_dir: Path) -> Path:
    d = cache_dir / SORTFORMER_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_epcontext_wrapper(cache_dir: Path) -> Path | None:
    """Generate the tiny EPContext-wrapper ONNX (sortformer_ctx.onnx) that points
    ORT-QNN at sortformer.bin. The AI Hub qnn_context_binary target ships ONLY the raw
    .bin (not an ONNX protobuf), so ORT cannot load it directly; this wrapper is the
    loadable model. Deterministic from the known I/O signature — regenerated whenever
    the .bin is present but the wrapper is missing. Returns the wrapper path, or None
    if no .bin. The context was compiled with --truncate_64bit_io, so its *_lengths
    I/O are int32.

    Needs the `onnx` package. The arm64 bundle ships the wrapper pre-made (so `onnx`
    isn't required at runtime); this generator is the fallback when only the .bin is
    present. Raises ImportError if `onnx` is absent and no pre-made wrapper exists."""
    import onnx
    from onnx import helper, TensorProto

    d = sortformer_model_dir(cache_dir)
    binf = d / "sortformer.bin"
    if not binf.is_file():
        return None
    wrap = d / "sortformer_ctx.onnx"

    F, I32 = TensorProto.FLOAT, TensorProto.INT32
    inputs = [("chunk", F, [1, FEED, N_MELS]), ("chunk_lengths", I32, [1]),
              ("spkcache", F, [1, SPKCACHE_LEN, EMB]), ("spkcache_lengths", I32, [1]),
              ("fifo", F, [1, FIFO_LEN, EMB]), ("fifo_lengths", I32, [1])]
    chunk_out = FEED // SUB
    out_t = SPKCACHE_LEN + FIFO_LEN + chunk_out
    outputs = [("spkcache_fifo_chunk_preds", F, [1, out_t, NSPK]),
               ("chunk_pre_encode_embs", F, [1, chunk_out, EMB]),
               ("chunk_pre_encode_lengths", I32, [1])]
    node = helper.make_node(
        "EPContext", inputs=[n for n, _, _ in inputs],
        outputs=[n for n, _, _ in outputs], name="sortformer_ctx",
        domain="com.microsoft", embed_mode=0, ep_cache_context="sortformer.bin",
        source="Qnn", ep_sdk_version="")   # source MUST be "Qnn" for ORT-QNN to claim it
    g = helper.make_graph(
        [node], "sortformer_ctx",
        [helper.make_tensor_value_info(n, t, s) for n, t, s in inputs],
        [helper.make_tensor_value_info(n, t, s) for n, t, s in outputs])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17),
                                            helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 10
    onnx.save(m, str(wrap))
    return wrap


def _npu_model_path(cache_dir: Path) -> Path | None:
    """The loadable NPU model is the EPContext-wrapper ONNX (`sortformer_ctx.onnx`),
    which references the raw QNN context binary (`sortformer.bin`) sitting next to it.
    onnxruntime-qnn CANNOT load the raw .bin directly (it is not an ONNX protobuf) —
    it loads the tiny wrapper, which points the HTP at the .bin. Both files must be
    present; the wrapper is what we hand to InferenceSession."""
    d = sortformer_model_dir(cache_dir)
    wrap = d / "sortformer_ctx.onnx"
    binf = d / "sortformer.bin"
    if wrap.is_file() and binf.is_file():
        return wrap
    return None


def _binary_path(cache_dir: Path) -> Path | None:
    # kept for the download-diagnostics; the loadable model is the wrapper.
    return _npu_model_path(cache_dir)


def _float_onnx_path(cache_dir: Path) -> Path | None:
    d = sortformer_model_dir(cache_dir)
    p = d / "sortformer.onnx"
    return p if p.is_file() else None


def sortformer_models_present(cache_dir: Path) -> bool:
    return _npu_model_path(cache_dir) is not None or _float_onnx_path(cache_dir) is not None


# ---------- QNN EP registration (onnxruntime-qnn 2.x plugin EP) ----------
def _ensure_qnn_registered():
    global _qnn_registered, _qnn_htp_path
    if _qnn_registered:
        return
    import onnxruntime as ort
    import onnxruntime_qnn as _qnn

    ort.register_execution_provider_library(_QNN_EP, _qnn.get_library_path())
    _qnn_htp_path = _qnn.get_qnn_htp_path()
    _qnn_registered = True


def qnn_npu_available() -> bool:
    try:
        _ensure_qnn_registered()
        import onnxruntime as ort

        return any(
            d.ep_name == _QNN_EP for d in ort.get_ep_devices()
        )
    except Exception:
        return False


def _qnn_htp_device():
    import onnxruntime as ort

    for d in ort.get_ep_devices():
        if d.ep_name == _QNN_EP:
            return d
    raise RuntimeError("No QNN HTP device found")


def _is_epcontext_model(model_path: Path) -> bool:
    """True if the ONNX is an EPContext wrapper (a QNN context binary reference),
    which must run on the QNN EP, not the CPU. Detected by the wrapper name or an
    EPContext node in the graph."""
    if model_path.name == "sortformer_ctx.onnx":
        return True
    try:
        import onnx

        m = onnx.load(str(model_path), load_external_data=False)
        return any(n.op_type == "EPContext" for n in m.graph.node)
    except Exception:
        return False


def _make_session(model_path: Path):
    """QNN session for the EPContext wrapper (-> .bin on the NPU); plain CPU session
    for the float .onnx."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if _is_epcontext_model(model_path):
        _ensure_qnn_registered()
        so.add_provider_for_devices([_qnn_htp_device()], {"backend_path": _qnn_htp_path})
        return ort.InferenceSession(str(model_path), sess_options=so)
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


# ---------- mel featurizer ----------
def _build_mel():
    import numpy as np

    def hz_to_mel(hz):
        f_sp = 200.0 / 3.0; mlh = 1000.0; mlm = mlh / f_sp; ls = 0.06875177742094912
        hz = np.atleast_1d(np.asarray(hz, np.float64)); out = hz / f_sp; m = hz >= mlh
        out[m] = mlm + np.log(hz[m] / mlh) / ls
        return out

    def mel_to_hz(mel):
        f_sp = 200.0 / 3.0; mlh = 1000.0; mlm = mlh / f_sp; ls = 0.06875177742094912
        mel = np.atleast_1d(np.asarray(mel, np.float64)); out = mel * f_sp; m = mel >= mlm
        out[m] = mlh * np.exp((mel[m] - mlm) * ls)
        return out

    bins = N_FFT // 2 + 1; fmax = SR / 2.0
    lo = hz_to_mel(0.0); hi = hz_to_mel(fmax)
    pts = mel_to_hz(lo + (hi - lo) * np.arange(N_MELS + 2) / (N_MELS + 1))
    ff = np.arange(bins) * SR / N_FFT; fd = np.diff(pts)
    fb = np.zeros((N_MELS, bins), np.float32)
    for i in range(N_MELS):
        lower = (ff - pts[i]) / fd[i]; upper = (pts[i + 2] - ff) / fd[i + 1]
        fb[i] = np.maximum(0.0, np.minimum(lower, upper))
    fb *= (2.0 / (pts[2:N_MELS + 2] - pts[0:N_MELS]))[:, None].astype(np.float32)
    win = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)  # periodic hann
    return fb, win


def _extract_mel(audio):
    import numpy as np

    fb, hann = _CACHE.setdefault("mel", _build_mel())
    a = np.asarray(audio, np.float32)
    pe = np.empty_like(a); pe[0] = a[0]; pe[1:] = a[1:] - PREEMPH * a[:-1]
    pad = N_FFT // 2
    padded = np.concatenate([np.zeros(pad, np.float32), pe, np.zeros(pad, np.float32)])
    off = (N_FFT - WIN) // 2; fw = np.zeros(N_FFT, np.float32); fw[off:off + WIN] = hann
    nf = (len(padded) - N_FFT) // HOP + 1
    spec = np.empty((N_FFT // 2 + 1, nf), np.float32)
    for f in range(nf):
        sp = np.fft.rfft(padded[f * HOP:f * HOP + N_FFT] * fw, N_FFT)
        spec[:, f] = (sp.real ** 2 + sp.imag ** 2).astype(np.float32)
    logmel = np.log((fb @ spec) + LOG_ZERO_GUARD)
    return logmel.T[None, :, :].astype(np.float32)


# ---------- diarizer ----------
class _SortformerRunner:
    def __init__(self, session):
        import numpy as np

        self.np = np
        self.s = session
        ins = {i.name: i for i in session.get_inputs()}
        self.static = isinstance(ins["spkcache"].shape[1], int)
        # The compiled QNN binary is built with --truncate_64bit_io, so its
        # *_lengths inputs are int32, not int64 (the float ONNX keeps int64).
        # Feed the dtype the session actually declares, else QNN rejects the run
        # ("Cannot assign data from unexpected type. Expected int32, got int64.").
        self.len_dtype = np.int32 if "int32" in ins["chunk_lengths"].type else np.int64
        self._reset()

    def _reset(self):
        np = self.np
        self.spk = np.zeros((1, 0, EMB), np.float32)
        self.fifo = np.zeros((1, 0, EMB), np.float32)
        self.fifo_preds = np.zeros((1, 0, NSPK), np.float32)
        self.spk_preds = None
        self.mean_sil_emb = np.zeros((1, EMB), np.float32)
        self.n_sil_frames = 0

    def _pad(self, arr, L, dim):
        np = self.np
        out = np.zeros((1, L, dim), np.float32); v = min(arr.shape[1], L)
        if v > 0:
            out[0, :v, :] = arr[0, :v, :]
        return out, v

    def _step(self, chunk_feat, cur):
        np = self.np
        sl = self.spk.shape[1]; fl = self.fifo.shape[1]
        if self.static:
            spk_in, spk_v = self._pad(self.spk, SPKCACHE_LEN, EMB)
            fifo_in, fifo_v = self._pad(self.fifo, FIFO_LEN, EMB)
        else:
            spk_in, spk_v = self.spk, sl
            fifo_in, fifo_v = self.fifo, fl
        ld = self.len_dtype
        feeds = {
            'chunk': chunk_feat.astype(np.float32), 'chunk_lengths': np.array([cur], ld),
            'spkcache': spk_in if spk_in.shape[1] > 0 else np.zeros((1, 0, EMB), np.float32),
            'spkcache_lengths': np.array([spk_v], ld),
            'fifo': fifo_in if fifo_in.shape[1] > 0 else np.zeros((1, 0, EMB), np.float32),
            'fifo_lengths': np.array([fifo_v], ld),
        }
        preds, new_embs, _ = self.s.run(None, feeds)
        valid = -(-cur // SUB); keep = min(CHUNK_LEN, valid)
        chunk_preds = preds[0, sl + fl:sl + fl + keep, :].copy()
        chunk_embs = new_embs[0, :keep, :].copy()
        self.fifo = chunk_embs[None] if self.fifo.shape[1] == 0 else np.concatenate([self.fifo, chunk_embs[None]], 1)
        if fl > 0:
            self.fifo_preds = np.concatenate([preds[0, sl:sl + fl, :], chunk_preds], 0)[None]
        else:
            self.fifo_preds = chunk_preds[None]
        if self.fifo.shape[1] > FIFO_LEN:
            pop = max(CHUNK_LEN, max(0, valid - FIFO_LEN) + fl); pop = min(pop, self.fifo.shape[1])
            pe = self.fifo[:, :pop, :]; pp = self.fifo_preds[:, :pop, :]
            self._update_silence_profile(pe, pp)
            self.fifo = self.fifo[:, pop:, :]; self.fifo_preds = self.fifo_preds[:, pop:, :]
            self.spk = np.concatenate([self.spk, pe], 1)
            if self.spk_preds is not None:
                self.spk_preds = np.concatenate([self.spk_preds, pp], 1)
            if self.spk.shape[1] > SPKCACHE_LEN:
                if self.spk_preds is None:
                    # initialize cache preds from this output's spkcache region + popped
                    init = preds[0, :sl, :][None]
                    self.spk_preds = np.concatenate([init, pp], 1)
                self._compress_spkcache()
        return chunk_preds

    # ---- NeMo smart speaker-cache compression (port of parakeet-rs) ----
    def _update_silence_profile(self, embs, preds):
        np = self.np
        p2 = preds[0]
        for t in range(p2.shape[0]):
            if p2[t].sum() < SIL_THRESHOLD:
                emb = embs[0, t]
                old = self.mean_sil_emb[0] * self.n_sil_frames
                self.n_sil_frames += 1
                self.mean_sil_emb[0] = (old + emb) / self.n_sil_frames

    def _compress_spkcache(self):
        np = self.np
        if self.spk_preds is None:
            return
        n = self.spk.shape[1]
        per_spk = SPKCACHE_LEN // NSPK
        if per_spk <= SPKCACHE_SIL_FRAMES_PER_SPK:
            self.spk = self.spk[:, :SPKCACHE_LEN, :]
            self.spk_preds = self.spk_preds[:, :SPKCACHE_LEN, :]
            return
        per = per_spk - SPKCACHE_SIL_FRAMES_PER_SPK
        strong = int(per * STRONG_BOOST_RATE); weak = int(per * WEAK_BOOST_RATE)
        min_pos = int(per * MIN_POS_SCORES_RATE)
        cp = self.spk_preds[0]                          # (n, NSPK)
        scores = self._log_pred_scores(cp)
        scores = self._disable_low(cp, scores, min_pos)
        scores = self._boost_topk(scores, strong, 2.0)
        scores = self._boost_topk(scores, weak, 1.0)
        if SPKCACHE_SIL_FRAMES_PER_SPK > 0:
            pad = np.full((n + SPKCACHE_SIL_FRAMES_PER_SPK, NSPK), -np.inf, np.float32)
            pad[:n] = scores
            pad[n:] = np.inf
            scores = pad
        idxs, disabled = self._topk_indices(scores, n)
        self._gather(idxs, disabled)

    def _log_pred_scores(self, preds):
        np = self.np
        p = np.maximum(preds, PRED_SCORE_THRESHOLD)
        log_1p = np.log(np.maximum(1.0 - p, PRED_SCORE_THRESHOLD))
        sum_log_1p = log_1p.sum(1, keepdims=True)
        return np.log(p) - log_1p + sum_log_1p - np.log(0.5)

    def _disable_low(self, preds, scores, min_pos):
        np = self.np
        pos_count = (scores > 0.0).sum(0)
        out = scores.copy()
        speech = preds > 0.5
        out[~speech] = -np.inf
        drop = speech & (scores <= 0.0) & (pos_count[None, :] >= min_pos)
        out[drop] = -np.inf
        return out

    def _boost_topk(self, scores, k, scale):
        np = self.np
        if k <= 0:
            return scores
        out = scores.copy()
        for s in range(NSPK):
            col = out[:, s]
            order = np.argsort(-col)[:min(k, len(col))]
            for t in order:
                if out[t, s] != -np.inf:
                    out[t, s] -= scale * np.log(0.5)
        return out

    def _topk_indices(self, scores, n_no_sil):
        np = self.np
        n = scores.shape[0]
        flat = scores.T.reshape(-1)               # speaker-major: s*n + t
        order = np.argsort(-flat)[:SPKCACHE_LEN]
        topk = [(MAX_INDEX if flat[i] == -np.inf else int(i)) for i in order]
        topk.sort()
        frame_idx = [0] * SPKCACHE_LEN
        disabled = [False] * SPKCACHE_LEN
        for i, fi in enumerate(topk):
            if fi == MAX_INDEX:
                disabled[i] = True
            else:
                f = fi % n
                if f >= n_no_sil:
                    disabled[i] = True
                else:
                    frame_idx[i] = f
        return frame_idx, disabled

    def _gather(self, idxs, disabled):
        np = self.np
        new_e = np.zeros((1, SPKCACHE_LEN, EMB), np.float32)
        new_p = np.zeros((1, SPKCACHE_LEN, NSPK), np.float32)
        cp = self.spk_preds
        for i, (idx, dis) in enumerate(zip(idxs, disabled)):
            if i >= SPKCACHE_LEN:
                break
            if dis:
                new_e[0, i] = self.mean_sil_emb[0]
            elif idx < self.spk.shape[1]:
                new_e[0, i] = self.spk[0, idx]
                new_p[0, i] = cp[0, idx]
        self.spk = new_e
        self.spk_preds = new_p

    def process(self, feats):
        np = self.np
        total = feats.shape[1]; nchunks = max(1, -(-total // CHUNK_STRIDE)); out = []
        for ci in range(nchunks):
            st = ci * CHUNK_STRIDE; en = min(st + FEED, total); cur = en - st
            cf = feats[:, st:en, :]
            if cur < FEED:
                pad = np.zeros((1, FEED, N_MELS), np.float32); pad[:, :cur, :] = cf; cf = pad
            out.append(self._step(cf, cur))
        return np.concatenate(out, 0) if out else np.zeros((0, NSPK), np.float32)

    def process_warmed(self, feats):
        """Static-graph warm-up: prepend WARMUP_SEC of the clip's own start (in the
        feature domain), process the whole thing so the caches fill before the real
        audio, then drop the warm-up frames. Falls back to plain process() if the
        clip is shorter than the warm-up. Only meaningful for the static/NPU path."""
        np = self.np
        total = feats.shape[1]
        warm_feat_frames = WARMUP_FRAMES * SUB      # mel frames for the pre-roll
        if total <= warm_feat_frames:
            return self.process(feats)              # too short to warm; run as-is
        pre = feats[:, :warm_feat_frames, :]
        warmed = np.concatenate([pre, feats], axis=1)
        preds = self.process(warmed)
        # drop the pre-roll's output frames (one output frame per SUB mel frames)
        drop = warm_feat_frames // SUB              # == WARMUP_FRAMES
        return preds[drop:drop + -(-total // SUB)]


def _median(preds, w, np):
    if w <= 1:
        return preds
    h = w // 2; f = preds.copy()
    for s in range(NSPK):
        for t in range(preds.shape[0]):
            a = max(0, t - h); b = min(preds.shape[0], t + h + 1)
            f[t, s] = np.median(preds[a:b, s])
    return f


def _binarize(preds):
    spf = FRAME_DUR * SR
    po = int(CFG['pad_onset'] * SR); pf = int(CFG['pad_offset'] * SR)
    mdo = int(CFG['min_duration_on'] * SR); mdoff = int(CFG['min_duration_off'] * SR)
    segs = []
    for spk in range(NSPK):
        tmp = []; inseg = False; start = 0
        for t in range(preds.shape[0]):
            p = preds[t, spk]
            if p >= CFG['onset'] and not inseg:
                inseg = True; start = t
            elif p < CFG['offset'] and inseg:
                inseg = False
                s = max(0, int(start * spf) - po); e = int(t * spf) + pf
                if e - s >= mdo:
                    tmp.append([s, e, spk])
        if inseg:
            s = max(0, int(start * spf) - po); e = int(preds.shape[0] * spf) + pf
            if e - s >= mdo:
                tmp.append([s, e, spk])
        if len(tmp) > 1:
            merged = [tmp[0]]
            for seg in tmp[1:]:
                if seg[0] - merged[-1][1] < mdoff:
                    merged[-1][1] = seg[1]
                else:
                    merged.append(seg)
            segs += merged
        else:
            segs += tmp
    segs.sort(key=lambda x: x[0])
    return segs


def _resample_linear(audio, sr_in, sr_out, np):
    if sr_in == sr_out:
        return audio
    n_out = int(math.floor(len(audio) * sr_out / sr_in))
    x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def sortformer_available(cache_dir: Path) -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return sortformer_models_present(cache_dir)


def get_runner(cache_dir: Path) -> tuple[_SortformerRunner, str]:
    """Cached runner. Prefers the compiled .bin (NPU); falls back to float .onnx (CPU)."""
    if "runner" in _CACHE:
        return _CACHE["runner"], _CACHE["backend"]
    binp = _binary_path(cache_dir)
    if binp is not None and qnn_npu_available():
        session = _make_session(binp)
        backend = "QNN / Hexagon NPU"
    else:
        onnxp = _float_onnx_path(cache_dir)
        if onnxp is None:
            raise RuntimeError("No Sortformer NPU binary and no float ONNX fallback present.")
        session = _make_session(onnxp)
        backend = "float ONNX / CPU (NPU unavailable)"
    runner = _SortformerRunner(session)
    _CACHE["runner"] = runner
    _CACHE["backend"] = backend
    return runner, backend


def _read_wav_mono(input_path: str, np):
    """Load first channel as float32 mono + sample rate. Prefers soundfile (the
    worker's other diarize backends use it); falls back to scipy.io.wavfile so the
    module has no hard soundfile dependency."""
    try:
        import soundfile as sf

        audio, sr = sf.read(input_path, dtype="float32", always_2d=True)
        return audio[:, 0], sr
    except ImportError:
        import scipy.io.wavfile as wf

        sr, data = wf.read(input_path)
        if data.dtype == np.int16:
            a = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            a = data.astype(np.float32) / 2147483648.0
        else:
            a = data.astype(np.float32)
        if a.ndim > 1:
            a = a[:, 0]
        return a, sr


def diarize_audio_sortformer(input_path: str, cache_dir: Path) -> dict[str, Any]:
    """Diarize with Sortformer on the NPU (or CPU float fallback). Muesli schema."""
    import numpy as np

    warnings: list[str] = []
    started_at = time.perf_counter()
    try:
        if not sortformer_models_present(cache_dir):
            warnings += [
                "Diarization skipped: Sortformer models not downloaded.",
                "Fetch them first via download_model(kind='diarize-sortformer-npu').",
            ]
            return _empty(warnings)

        runner, backend = get_runner(cache_dir)
        model_ms = (time.perf_counter() - started_at) * 1000

        audio, sr = _read_wav_mono(input_path, np)
        audio = _resample_linear(audio, sr, SR, np)
        n = len(audio)

        infer_started_at = time.perf_counter()
        runner._reset()
        feats = _extract_mel(audio)
        # Static/frozen (NPU) graphs need the cache warm-up; the dynamic ONNX does not.
        if runner.static:
            preds = runner.process_warmed(feats)
        else:
            preds = runner.process(feats)
        preds = _median(preds, CFG['median_window'], np)
        infer_ms = (time.perf_counter() - infer_started_at) * 1000

        # dense 1-based labels in order of first appearance
        label_of: dict[int, str] = {}
        segments = []
        for s, e, spk in _binarize(preds):
            e = min(e, n)
            if e <= s:
                continue
            if spk not in label_of:
                label_of[spk] = f"Speaker {len(label_of) + 1}"
            segments.append({
                "id": f"diarize_{len(segments)}",
                "speaker": label_of[spk],
                "startMs": int(s * 1000 // SR),
                "endMs": int(e * 1000 // SR),
                "text": "",
            })

        total_ms = (time.perf_counter() - started_at) * 1000
        warnings += [
            "ASR engine: diarization",
            f"Diarization backend: Sortformer / {backend}",
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
        warnings += ["Diarization failed (Sortformer).", f"Error: {exc}"]
        return _empty(warnings)


def _empty(warnings: list[str]) -> dict[str, Any]:
    return {
        "transcriptText": "",
        "detectedLanguage": "en",
        "durationMs": 0,
        "segments": [],
        "warnings": warnings,
    }
