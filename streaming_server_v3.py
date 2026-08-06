"""
streaming_server_v3.py

Benchmarking / debugging streaming ASR server built on top of the same
Conformer-RNNT + CacheFeatureBufferer pipeline used by streaming_server_v2.py.

New in v3:
  - Word-level transcript stabilization (no character-level flickering).
  - Per-step benchmarking: feature-extraction ms, inference ms, total ms, RTF,
    actual chunk-interval ms.
  - Running statistics: avg / median / std / min / max / p95 for every metric.
  - Periodic benchmark report printed every BENCH_PRINT_EVERY_N_CHUNKS chunks.
  - Latency breakdown (chunk-wait / feature / inference / JSON / WS-send)
    printed when DEBUG_TIMING=True.
  - Independent debug flags: DEBUG_TIMING, DEBUG_FULL_HYP, DEBUG_CACHE,
    DEBUG_PARTIALS, DEBUG_WORD_STABILIZER, DEBUG_CPU, DEBUG_FEATURES.
  - Startup summary: CPU info, torch threading, model path, att-context,
    streaming config, chunk duration.
  - Graceful error recovery: WebSocket never crashes; tracebacks logged.
  - WebSocket protocol identical to v2 (partial / final / ready / error).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import asyncio
import json
import math
import platform
import statistics
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import nemo.collections.asr as nemo_asr

# ==============================================================================
# CONFIGURATION  (edit these constants before running)
# ==============================================================================

MODEL_PATH: str = "D:/FineTuning/final_after_150steps.nemo"
TARGET_LANG: str = "ne-NP"
DEVICE: str = "cpu"
THREADS: Optional[int] = None          # None -> use os.cpu_count()

# Switchable latency/accuracy profiles for the streaming encoder's
# att_context_size. Right-context frame options control lookahead
# (0 / 240 / 480ms for low/medium/high, given subsampling_factor=8 and
# window_stride=0.01s). Change ATT_CONTEXT_PROFILE to benchmark a different
# profile; ATT_CONTEXT_SIZE is derived from it below.
ATT_CONTEXT_PROFILES: dict[str, List[int]] = {
    "low": [56, 0],
    "high-low": [56, 1],
    "low-medium": [56, 3],
    "medium": [56, 6],
    "high": [56, 13],
}
ATT_CONTEXT_PROFILE: str = "low-medium"    # "low" | "low-medium" | "medium" | "high"
ATT_CONTEXT_SIZE: List[int] = ATT_CONTEXT_PROFILES[ATT_CONTEXT_PROFILE]

HOST: str = "0.0.0.0"
PORT: int = 8000

SAMPLE_RATE: int = 16000
LOG_MEL_ZERO: float = -16.635         # CacheFeatureBufferer default fill value

# How many chunks between each benchmark report line.
BENCH_PRINT_EVERY_N_CHUNKS: int = 10

# ==============================================================================
# DEBUG FLAGS  (each independently enables its log section)
# ==============================================================================

DEBUG_TIMING: bool = False        # per-step latency breakdown
DEBUG_FULL_HYP: bool = False      # raw hyp / stable prefix / delta / emitted
DEBUG_CACHE: bool = False         # cache state shapes after each step
DEBUG_PARTIALS: bool = False      # every partial text that passes the filter
DEBUG_WORD_STABILIZER: bool = False  # word-level stabilizer internals
DEBUG_CPU: bool = False           # thread / interop thread counts each step
DEBUG_FEATURES: bool = False      # feature buffer shape / stats each step
DEBUG_INVARIANTS: bool = True     # check stabilizer invariants and print violations

# ==============================================================================
# THREAD SETUP
# ==============================================================================

_actual_threads: int = THREADS if THREADS is not None else (os.cpu_count() or 1)
torch.set_num_threads(_actual_threads)

# ==============================================================================
# STARTUP: CPU INFORMATION
# ==============================================================================

def _print_cpu_info() -> None:
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
        phys_str = str(phys)
    except ImportError:
        phys_str = "unavailable (install psutil)"

    print("=" * 62)
    print("  Streaming ASR Server v3 — Startup Summary")
    print("=" * 62)
    print(f"  Platform processor : {platform.processor()}")
    print(f"  Logical cores      : {os.cpu_count()}")
    print(f"  Physical cores     : {phys_str}")
    print(f"  Torch version      : {torch.__version__}")
    print(f"  Torch threads      : {torch.get_num_threads()} (set to {_actual_threads})")
    print(f"  Torch interop thds : {torch.get_num_interop_threads()}")
    print(f"  Device             : {DEVICE}")
    print(f"  Model path         : {MODEL_PATH}")
    print(f"  Target language    : {TARGET_LANG}")
    print(f"  ATT_CONTEXT_PROFILE: {ATT_CONTEXT_PROFILE} -> {ATT_CONTEXT_SIZE}")
    print("=" * 62)

_print_cpu_info()

# ==============================================================================
# MODEL LOAD
# ==============================================================================

print("[startup] Loading model ...")
model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location=DEVICE)
model.eval()
model.to(DEVICE)
print("[startup] Model loaded.")

if not getattr(model, "concat", False):
    raise RuntimeError(
        "model.concat is False -- the prompt-conditioning path in "
        "_apply_prompt_to_encoded will silently no-op. Check "
        "initialize_prompt_feature in the restored checkpoint."
    )

model.set_inference_prompt(TARGET_LANG)

# --- Prompt verification (checked once at startup, re-checked every session) ---
_expected_prompt_index = model.cfg.model_defaults.prompt_dictionary.get(TARGET_LANG)
_active_prompt_index = getattr(model, "_inference_prompt_index", None)

print(f"[startup] prompt_dictionary[{TARGET_LANG!r}] = {_expected_prompt_index}")
print(f"[startup] active _inference_prompt_index    = {_active_prompt_index}")

if _expected_prompt_index is None or _active_prompt_index != _expected_prompt_index:
    raise RuntimeError(
        f"PROMPT MISMATCH at startup: expected index {_expected_prompt_index} for "
        f"TARGET_LANG={TARGET_LANG!r} but model._inference_prompt_index = "
        f"{_active_prompt_index}. Refusing to start -- this would silently leak "
        f"into another language's prompt (e.g. hi-IN)."
    )
print(f"[startup] Prompt verified OK: {TARGET_LANG!r} -> index {_active_prompt_index}")

if hasattr(model.encoder, "set_default_att_context_size"):
    model.encoder.set_default_att_context_size(att_context_size=ATT_CONTEXT_SIZE)
    print(f"[startup] att_context_size set to {ATT_CONTEXT_SIZE}")
else:
    raise RuntimeError(
        "model.encoder has no set_default_att_context_size -- this checkpoint "
        "may not support multiple latency profiles."
    )

model.encoder.setup_streaming_params()

# ==============================================================================
# CHUNK CONSTANTS (derived from streaming_cfg after att_context is applied)
# ==============================================================================

def _compute_chunk_constants() -> tuple[float, float, int]:
    window_stride = model.cfg.preprocessor.window_stride

    shift_frames = model.encoder.streaming_cfg.shift_size
    shift_frames = shift_frames[1] if isinstance(shift_frames, list) else shift_frames

    precache_frames = model.encoder.streaming_cfg.pre_encode_cache_size
    precache_frames = precache_frames[1] if isinstance(precache_frames, list) else precache_frames

    chunk_secs = shift_frames * window_stride
    buffer_secs = (shift_frames + precache_frames) * window_stride
    new_samples = int(chunk_secs * SAMPLE_RATE)
    return chunk_secs, buffer_secs, new_samples


CHUNK_SIZE_IN_SECS, BUFFER_SIZE_IN_SECS, NEW_SAMPLES_PER_STEP = _compute_chunk_constants()
CHUNK_MS: float = NEW_SAMPLES_PER_STEP / SAMPLE_RATE * 1000.0

_drop_extra_pre_encoded = model.encoder.streaming_cfg.drop_extra_pre_encoded

print(
    f"[startup] streaming_cfg={model.encoder.streaming_cfg}\n"
    f"[startup] chunk={CHUNK_SIZE_IN_SECS:.3f}s  "
    f"buffer={BUFFER_SIZE_IN_SECS:.3f}s  "
    f"new_samples/step={NEW_SAMPLES_PER_STEP}  "
    f"chunk_duration={CHUNK_MS:.0f}ms"
)
print("=" * 62)


def verify_prompt_or_warn() -> int:
    """Re-check the model's active inference prompt index.

    The prompt index is global model state (set once via
    model.set_inference_prompt), so this catches the case where something
    else in the process (e.g. a concurrent session) mutated it away from
    TARGET_LANG between startup and this session/request.
    """
    active = getattr(model, "_inference_prompt_index", None)
    if active != _expected_prompt_index:
        print(
            f"[WARN] PROMPT LEAK DETECTED: expected {TARGET_LANG!r} "
            f"(index {_expected_prompt_index}) but active index = {active}. "
            f"Forcing model back to {TARGET_LANG!r}.",
            flush=True,
        )
        model.set_inference_prompt(TARGET_LANG)
        active = getattr(model, "_inference_prompt_index", None)
    else:
        print(
            f"[session] prompt verified: {TARGET_LANG!r} -> index {active}",
            flush=True,
        )
    return active


# Threading lock: inference runs in asyncio.to_thread (OS threads), not the loop.
inference_lock = threading.Lock()

app = FastAPI()

# ==============================================================================
# BENCHMARKING ACCUMULATOR
# ==============================================================================

@dataclass
class _RunningStats:
    """Accumulates float samples; computes descriptive stats lazily."""
    _samples: List[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        self._samples.append(value)

    def reset(self) -> None:
        self._samples.clear()

    @property
    def count(self) -> int:
        return len(self._samples)

    def _require(self) -> List[float]:
        if not self._samples:
            raise ValueError("no samples")
        return self._samples

    def avg(self) -> float:
        s = self._require()
        return sum(s) / len(s)

    def median(self) -> float:
        return statistics.median(self._require())

    def std(self) -> float:
        s = self._require()
        return statistics.pstdev(s) if len(s) > 1 else 0.0

    def minimum(self) -> float:
        return min(self._require())

    def maximum(self) -> float:
        return max(self._require())

    def p95(self) -> float:
        s = sorted(self._require())
        idx = max(0, math.ceil(0.95 * len(s)) - 1)
        return s[idx]


@dataclass
class BenchmarkAccumulator:
    feat_ms: _RunningStats = field(default_factory=_RunningStats)
    infer_ms: _RunningStats = field(default_factory=_RunningStats)
    total_ms: _RunningStats = field(default_factory=_RunningStats)
    rtf: _RunningStats = field(default_factory=_RunningStats)
    chunk_interval_ms: _RunningStats = field(default_factory=_RunningStats)
    _chunk_count: int = 0
    _last_inference_ts: Optional[float] = None

    def record_step(self, feat_ms: float, infer_ms: float, total_ms: float) -> None:
        self.feat_ms.record(feat_ms)
        self.infer_ms.record(infer_ms)
        self.total_ms.record(total_ms)
        self.rtf.record(total_ms / CHUNK_MS)

        now = time.perf_counter()
        if self._last_inference_ts is not None:
            self.chunk_interval_ms.record((now - self._last_inference_ts) * 1000.0)
        self._last_inference_ts = now
        self._chunk_count += 1

    def should_print(self) -> bool:
        return self._chunk_count > 0 and self._chunk_count % BENCH_PRINT_EVERY_N_CHUNKS == 0

    def print_report(self) -> None:
        def _row(label: str, rs: _RunningStats) -> str:
            if rs.count == 0:
                return f"  {label:<20}  n/a"
            return (
                f"  {label:<20}"
                f"  avg={rs.avg():8.2f}"
                f"  med={rs.median():8.2f}"
                f"  std={rs.std():7.2f}"
                f"  min={rs.minimum():8.2f}"
                f"  max={rs.maximum():8.2f}"
                f"  p95={rs.p95():8.2f}"
            )

        sep = "=" * 92
        print(sep)
        print(
            f"  Benchmark report  "
            f"(chunks={self._chunk_count}  chunk_dur={CHUNK_MS:.0f}ms)"
        )
        print(
            f"  {'Metric':<20}  {'avg':>9}  {'med':>9}  {'std':>8}  "
            f"{'min':>9}  {'max':>9}  {'p95':>9}"
        )
        print("-" * 92)
        print(_row("Feature (ms)", self.feat_ms))
        print(_row("Inference (ms)", self.infer_ms))
        print(_row("Total (ms)", self.total_ms))
        print(_row("RTF", self.rtf))
        if self.chunk_interval_ms.count:
            print(_row("ChunkInterval (ms)", self.chunk_interval_ms))
        print(sep)


# ==============================================================================
# RAW-AUDIO RING BUFFER
# ==============================================================================

class _RawAudioRingBuffer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float) -> None:
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.samples = torch.zeros(self.buffer_size, dtype=torch.float32)

    def update(self, audio: np.ndarray) -> None:
        audio_t = torch.from_numpy(audio)
        shift = audio_t.shape[0]
        if shift > self.buffer_size:
            raise ValueError(
                f"chunk ({shift} samples) exceeds ring buffer ({self.buffer_size} samples)"
            )
        self.samples[:-shift] = self.samples[shift:].clone()
        self.samples[-shift:] = audio_t.clone()


# ==============================================================================
# CACHE FEATURE BUFFERER (ported from NeMo voice-agent utils.py)
# ==============================================================================

class CacheFeatureBufferer:
    """Per-session feature extractor using a rolling raw-audio ring buffer.

    Ported from nemo/agents/voice_agent/pipecat/services/nemo/utils.py,
    scoped to CPU / this model's preprocessor config.
    """

    def __init__(self, preprocessor_cfg, device: str = "cpu") -> None:
        self.device = device
        self.n_feat = preprocessor_cfg.features
        self.timestep_duration = preprocessor_cfg.window_stride
        self.n_chunk_look_back = int(self.timestep_duration * SAMPLE_RATE)
        self.chunk_size = int(CHUNK_SIZE_IN_SECS * SAMPLE_RATE)

        self.sample_buffer = _RawAudioRingBuffer(SAMPLE_RATE, BUFFER_SIZE_IN_SECS)

        self.feature_buffer_len = int(BUFFER_SIZE_IN_SECS / self.timestep_duration)
        self.feature_chunk_len = int(CHUNK_SIZE_IN_SECS / self.timestep_duration)
        self.zero_val = LOG_MEL_ZERO if getattr(preprocessor_cfg, "log", True) else 0.0
        self.feature_buffer = torch.full(
            [self.n_feat, self.feature_buffer_len],
            self.zero_val,
            dtype=torch.float32,
            device=device,
        )

        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(device)

    def _update_feature_buffer(self, feat_chunk: torch.Tensor) -> None:
        self.feature_buffer[:, : -self.feature_chunk_len] = (
            self.feature_buffer[:, self.feature_chunk_len :].clone()
        )
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def _preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        audio_signal = audio_signal.unsqueeze(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(input_signal=audio_signal, length=audio_signal_len)
        return features.squeeze(0)

    def update(self, audio: np.ndarray) -> None:
        self.sample_buffer.update(audio)
        samples = self.sample_buffer.samples[-(self.n_chunk_look_back + self.chunk_size) :]
        features = self._preprocess(samples)
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])

    def get_feature_buffer(self) -> torch.Tensor:
        return self.feature_buffer.clone()


# ==============================================================================
# WORD-LEVEL TRANSCRIPT STABILIZER
# ==============================================================================

class WordStabilizer:
    """Stabilized-prefix delta emission at the word level.

    Maintains:
      _already_emitted   -- words already sent to client (never resent).
      _previous_snapshot -- last full hypothesis word list.

    Algorithm (each hypothesis):
      1. Split previous_snapshot and current text into word lists.
      2. Find longest common word prefix between them.
      3. Words in that prefix beyond _already_emitted are newly stable.
      4. Emit them; advance _already_emitted.
      5. Trailing (unstable) words are suppressed.

    finalize():
      Flush all remaining unemitted words unconditionally, UNLESS the model
      revised an already-emitted word mid-stream (possible at low right-
      context / lookahead), in which case _already_emitted is no longer
      trustworthy -- fall back to the caller-supplied hypothesis directly.
    """

    def __init__(self) -> None:
        self._already_emitted: List[str] = []
        self._previous_snapshot: List[str] = []
        self._corrupted: bool = False

    @staticmethod
    def _split(text: str) -> List[str]:
        return [w for w in text.split() if w]

    @staticmethod
    def _common_prefix_len(a: List[str], b: List[str]) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def update(self, hypothesis_text: str) -> Optional[str]:
        """Return newly stabilized words as a string, or None if nothing new."""
        current = self._split(hypothesis_text)

        # Corruption check: if the model revised an already-emitted word
        # (current no longer starts with _already_emitted), the stable-prefix
        # bookkeeping can no longer be trusted. Flag it; finalize() will fall
        # back to the raw hypothesis instead of trusting _already_emitted.
        already_n = len(self._already_emitted)
        if current[:already_n] != self._already_emitted:
            if not self._corrupted:
                print(
                    f"[stabilizer] WARN: hypothesis revised an already-emitted "
                    f"word -- stable-prefix state corrupted. "
                    f"emitted={self._already_emitted}  current={current}",
                    flush=True,
                )
            self._corrupted = True

        stable_len = self._common_prefix_len(self._previous_snapshot, current)

        if DEBUG_WORD_STABILIZER:
            print(
                f"[stabilizer] prev={self._previous_snapshot}  "
                f"curr={current}  stable_len={stable_len}  "
                f"emitted={self._already_emitted}"
            )

        self._previous_snapshot = current

        if stable_len <= already_n:
            return None

        new_words = current[already_n:stable_len]
        self._already_emitted.extend(new_words)
        delta = " ".join(new_words)

        if DEBUG_FULL_HYP:
            print(
                f"[hyp]  raw      = {hypothesis_text!r}\n"
                f"       stable   = {' '.join(current[:stable_len])!r}\n"
                f"       delta    = {delta!r}\n"
                f"       emitted  = {' '.join(self._already_emitted)!r}"
            )

        return delta if delta else None

    def finalize(self, hypothesis_text: str) -> str:
        """Flush all remaining words not yet emitted; return full transcript.

        If update() detected mid-stream corruption (the model revised an
        already-emitted word), _already_emitted is not trustworthy -- ignore
        it and return the caller-supplied hypothesis_text directly instead.
        """
        current = self._split(hypothesis_text)

        if self._corrupted:
            self._already_emitted = list(current)
            print(
                f"[stabilizer] finalize: corrupted state -- falling back to "
                f"hypothesis_text directly: {current}",
                flush=True,
            )
            return " ".join(current)

        already_n = len(self._already_emitted)
        remaining = current[already_n:]
        self._already_emitted.extend(remaining)

        if DEBUG_WORD_STABILIZER:
            print(
                f"[stabilizer] finalize: remaining={remaining}  "
                f"full={self._already_emitted}"
            )

        return " ".join(self._already_emitted)


# ==============================================================================
# HELPERS
# ==============================================================================

def pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
    int16 = np.frombuffer(raw, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


# ==============================================================================
# STREAMING SESSION
# ==============================================================================

class StreamingSession:
    def __init__(self) -> None:
        self.active_prompt_index = verify_prompt_or_warn()
        self._session_start_ts = time.perf_counter()

        self.feature_bufferer = CacheFeatureBufferer(model.cfg.preprocessor, device=DEVICE)
        self._pending: np.ndarray = np.empty(0, dtype=np.float32)

        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)

        self.previous_hypotheses = None
        self.finalized: bool = False
        self.transcript_text: str = ""          # last hypothesis from RNNT (any step)
        self.best_transcript_text: str = ""     # FIX: best (longest-word) real hypothesis
        self.stabilizer = WordStabilizer()
        self.bench = BenchmarkAccumulator()

    # ------------------------------------------------------------------
    # Core inference step
    # ------------------------------------------------------------------

    def _run_one_step(
        self, new_audio: np.ndarray, keep_all_outputs: bool,
        update_stabilizer: bool = True,
    ) -> tuple[Optional[str], Optional[str]]:
        """Run one encoder+decoder step.

        Args:
            update_stabilizer: If False, the hypothesis is recorded but the
                WordStabilizer is NOT updated. Set to False for silence-flush
                steps so that degraded silence hypotheses do not corrupt the
                stable-prefix state or overwrite best_transcript_text.

        Returns (full_hypothesis_text, delta_of_newly_stable_words).
        Either element may be None.
        """
        # --- Feature extraction timing ---
        t_feat_start = time.perf_counter()
        self.feature_bufferer.update(new_audio)
        features = self.feature_bufferer.get_feature_buffer()  # (n_feat, T)
        feat_ms = (time.perf_counter() - t_feat_start) * 1000.0

        if DEBUG_FEATURES:
            print(
                f"[features] shape={features.shape}  "
                f"mean={features.mean():.4f}  std={features.std():.4f}"
            )

        print("NEW_SAMPLES_PER_STEP =", NEW_SAMPLES_PER_STEP)
        print("CHUNK_SIZE_IN_SECS   =", CHUNK_SIZE_IN_SECS)
        print("features.shape[-1]   =", features.shape[-1])
        print("streaming_cfg        =", model.encoder.streaming_cfg)

        features = features.unsqueeze(0)                        # (1, n_feat, T)
        feature_lengths = torch.tensor([features.shape[-1]])

        # --- Inference timing ---
        t_infer_start = time.perf_counter()
        with inference_lock:
            with torch.inference_mode():
                (
                    _greedy_predictions,
                    _all_hyp,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    best_hyp,
                ) = model.conformer_stream_step(
                    processed_signal=features,
                    processed_signal_length=feature_lengths,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=keep_all_outputs,
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=None,   # CTC-only; unused for RNNT
                    drop_extra_pre_encoded=_drop_extra_pre_encoded,
                    return_transcription=True,
                    return_log_probs=False,
                )
        infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
        total_ms = feat_ms + infer_ms

        # --- Record benchmarks ---
        self.bench.record_step(feat_ms, infer_ms, total_ms)

        if DEBUG_TIMING:
            print(
                f"[timing] feat={feat_ms:.2f}ms  "
                f"infer={infer_ms:.2f}ms  "
                f"total={total_ms:.2f}ms  "
                f"RTF={total_ms / CHUNK_MS:.3f}"
            )

        if DEBUG_CPU:
            print(
                f"[cpu] threads={torch.get_num_threads()}  "
                f"interop={torch.get_num_interop_threads()}"
            )

        if DEBUG_CACHE:
            print(
                f"[cache] channel={self.cache_last_channel.shape}  "
                f"time={self.cache_last_time.shape}  "
                f"channel_len={self.cache_last_channel_len}"
            )

        if self.bench.should_print():
            self.bench.print_report()

        # --- Hypothesis processing ---
        self.previous_hypotheses = best_hyp

        energy = float(np.sqrt(np.mean(new_audio ** 2)))

        if best_hyp is None or len(best_hyp) == 0:
            print(
                f"[step] energy={energy:.5f}  text=None  y_seq_len=0  "
                f"feat={feat_ms:.1f}ms  infer={infer_ms:.1f}ms  "
                f"RTF={total_ms / CHUNK_MS:.3f}  "
                f"stabilizer_update={update_stabilizer}",
                flush=True,
            )
            return None, None

        hyp_text: str = best_hyp[0].text

        # FIX 1: Always record the raw transcript so we can see what the model
        # produced, but only promote to best_transcript_text when the hypothesis
        # is at least as long (word-count) as the current best.  Silence-flush
        # steps routinely produce shorter hypotheses as the beam empties; we
        # must not let those overwrite the best real transcript.
        self.transcript_text = hyp_text
        hyp_words = [w for w in hyp_text.split() if w]
        best_words = [w for w in self.best_transcript_text.split() if w]
        if len(hyp_words) >= len(best_words):
            self.best_transcript_text = hyp_text

        if DEBUG_INVARIANTS:
            if len(hyp_words) < len(best_words):
                print(
                    f"[invariant2] transcript shorter than best: "
                    f"hyp={hyp_text!r}  best={self.best_transcript_text!r}  "
                    f"update_stabilizer={update_stabilizer}"
                )

        print(
            f"[step] energy={energy:.5f}  text={hyp_text!r}  "
            f"best={self.best_transcript_text!r}  "
            f"y_seq_len={len(best_hyp[0].y_sequence)}  "
            f"feat={feat_ms:.1f}ms  infer={infer_ms:.1f}ms  "
            f"RTF={total_ms / CHUNK_MS:.3f}  "
            f"stabilizer_update={update_stabilizer}",
            flush=True,
        )

        # FIX 2: Only update the stabilizer during real streaming steps, not
        # during the silence-flush steps inside finalize().  Silence hypotheses
        # are shorter / degraded and would corrupt _previous_snapshot, making
        # subsequent stable_len calculations wrong.
        if not update_stabilizer:
            return hyp_text, None

        delta = self.stabilizer.update(hyp_text)

        if DEBUG_INVARIANTS:
            emitted_str = " ".join(self.stabilizer._already_emitted)
            if not hyp_text.startswith(emitted_str) and emitted_str:
                print(
                    f"[invariant1] ERROR: already_emitted is not a prefix of hyp!\n"
                    f"  emitted={emitted_str!r}\n"
                    f"  hyp    ={hyp_text!r}"
                )

        if DEBUG_PARTIALS and delta:
            print(f"[partial_delta] {delta!r}")

        return hyp_text, delta

    # ------------------------------------------------------------------
    # Public interface (synchronous -- called via asyncio.to_thread)
    # ------------------------------------------------------------------

    def process_chunk(self, raw_pcm_bytes: bytes) -> Optional[str]:
        """Accumulate raw PCM bytes; run encoder steps when enough audio is ready.

        Returns the current running emitted transcript (all stable words so far),
        or None when no new words were stabilized.
        """
        new_audio = pcm16_bytes_to_float32(raw_pcm_bytes)
        self._pending = np.concatenate([self._pending, new_audio])

        had_output = False

        while len(self._pending) >= NEW_SAMPLES_PER_STEP:
            step_audio = self._pending[:NEW_SAMPLES_PER_STEP]
            self._pending = self._pending[NEW_SAMPLES_PER_STEP:]
            _full, delta = self._run_one_step(step_audio, keep_all_outputs=False)
            if delta is not None:
                had_output = True

        if had_output:
            # Return everything emitted so far as the running partial transcript.
            return " ".join(self.stabilizer._already_emitted)
        return None

    def finalize(self) -> str:
        """Pad + flush remaining audio; return final complete transcript."""
        if self.finalized:
            return self.best_transcript_text
        self.finalized = True

        # Pad leftover audio to a full step.
        # update_stabilizer=True here because this is still real audio.
        if len(self._pending) > 0:
            pad_len = NEW_SAMPLES_PER_STEP - len(self._pending)
            if pad_len > 0:
                self._pending = np.concatenate(
                    [self._pending, np.zeros(pad_len, dtype=np.float32)]
                )
            self._run_one_step(
                self._pending[:NEW_SAMPLES_PER_STEP],
                keep_all_outputs=False,
                update_stabilizer=True,
            )
            self._pending = np.empty(0, dtype=np.float32)

        # Feed silence to flush trailing RNNT symbols.
        # FIX 3: update_stabilizer=False so silence-degraded hypotheses do NOT
        # corrupt _previous_snapshot or overwrite best_transcript_text.
        silence = np.zeros(NEW_SAMPLES_PER_STEP, dtype=np.float32)
        self._run_one_step(silence, keep_all_outputs=False, update_stabilizer=False)
        self._run_one_step(silence, keep_all_outputs=True,  update_stabilizer=False)

        # FIX 3 cont.: pass best_transcript_text (not silence-degraded
        # transcript_text) to the stabilizer so the flush is correct.
        if DEBUG_INVARIANTS:
            print(
                f"[finalize] transcript_text={self.transcript_text!r}  "
                f"best_transcript_text={self.best_transcript_text!r}  "
                f"already_emitted={self.stabilizer._already_emitted}"
            )

        final_text = self.stabilizer.finalize(self.best_transcript_text)
        self.best_transcript_text = final_text
        self.transcript_text = final_text

        total_transcription_ms = (time.perf_counter() - self._session_start_ts) * 1000.0

        print(
            f"[finalize] att_context_profile={ATT_CONTEXT_PROFILE} {ATT_CONTEXT_SIZE}  "
            f"prompt={TARGET_LANG!r}(index={self.active_prompt_index})  "
            f"total_transcription_ms={total_transcription_ms:.1f}  "
            f"final_text={final_text!r}",
            flush=True,
        )
        self.bench.print_report()
        return final_text


# ==============================================================================
# WEBSOCKET ENDPOINT
# ==============================================================================

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket) -> None:
    await websocket.accept()
    session = StreamingSession()

    # Signal the client that the model is loaded and ready.
    try:
        await websocket.send_json({"type": "ready"})
    except Exception:
        return

    try:
        while True:
            t_recv_start = time.perf_counter()
            message = await websocket.receive()
            recv_wait_ms = (time.perf_counter() - t_recv_start) * 1000.0

            # ---- Binary audio chunk ----
            if "bytes" in message and message["bytes"] is not None:
                raw_pcm = message["bytes"]

                if DEBUG_TIMING:
                    print(
                        f"[timing] recv_wait={recv_wait_ms:.2f}ms  "
                        f"payload={len(raw_pcm)}B"
                    )

                t_thread_start = time.perf_counter()
                partial_text = await asyncio.to_thread(session.process_chunk, raw_pcm)
                t_thread_ms = (time.perf_counter() - t_thread_start) * 1000.0

                if partial_text is not None:
                    payload = {"type": "partial", "text": partial_text}

                    t_json_start = time.perf_counter()
                    payload_str = json.dumps(payload)
                    json_ms = (time.perf_counter() - t_json_start) * 1000.0

                    t_send_start = time.perf_counter()
                    await websocket.send_text(payload_str)
                    send_ms = (time.perf_counter() - t_send_start) * 1000.0

                    if DEBUG_TIMING:
                        print(
                            f"[timing] thread={t_thread_ms:.2f}ms  "
                            f"json_ser={json_ms:.3f}ms  "
                            f"ws_send={send_ms:.3f}ms"
                        )

            # ---- END signal ----
            elif "text" in message and message["text"] == "END":
                try:
                    final_text = await asyncio.to_thread(session.finalize)
                except Exception:
                    tb = traceback.format_exc()
                    print(f"[error] finalize failed:\n{tb}")
                    try:
                        await websocket.send_json({"type": "error", "text": tb})
                    except Exception:
                        pass
                    await websocket.close()
                    return

                await websocket.send_json({"type": "final", "text": final_text})
                await websocket.close()
                break

    except WebSocketDisconnect:
        print("[ws] client disconnected", flush=True)
        try:
            session.finalize()
        except Exception:
            pass
    except Exception:
        tb = traceback.format_exc()
        print(f"[error] unhandled exception:\n{tb}")
        try:
            await websocket.send_json({"type": "error", "text": "Internal server error."})
            await websocket.close()
        except Exception:
            pass
        try:
            session.finalize()
        except Exception:
            pass