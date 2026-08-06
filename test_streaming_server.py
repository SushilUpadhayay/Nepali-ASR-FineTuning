"""
Real-time streaming ASR server for the fine-tuned Nemotron-3.5-ASR-Streaming-0.6B
(ne-NP) model.

Chunk sizing is derived from this model's actual streaming_cfg (not copied
from the reference service, which was tuned for a different model):
  - window_stride = 0.01s -> 160 raw samples / mel frame
  - steady-state shift_size = 32 frames -> 5120 raw samples (320ms) of NEW
    audio required per encoder step
  - steady-state pre_encode_cache_size = 9 frames -> 90ms of look-back
    context alongside each chunk
  - buffer_size_in_secs = 0.41 (41 total frames fed to the encoder per step)

No pseudo-streaming. No repeated model.transcribe() calls. No first-chunk
special-casing (the reference service doesn't special-case it either -- the
encoder's zero-initialized cache handles startup correctly on its own).
"""

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import asyncio
import threading
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import nemo.collections.asr as nemo_asr

torch.set_num_threads(os.cpu_count())

MODEL_PATH = "D:/FineTuning/final_after_150steps.nemo"
TARGET_LANG = "ne-NP"
SAMPLE_RATE = 16000
LOG_MEL_ZERO = -16.635  # matches CacheFeatureBufferer's default fill_value

app = FastAPI()

# ---------------------------------------------------------------------------
# Model: loaded once at process startup, shared across all sessions.
# ---------------------------------------------------------------------------
model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH, map_location="cpu")
model.eval()
model.to("cpu")

model.set_inference_prompt(TARGET_LANG)

if model.encoder.streaming_cfg is None:
    model.encoder.setup_streaming_params()

# --- Chunk sizing, computed from this model's actual streaming_cfg --------
_window_stride = model.cfg.preprocessor.window_stride  # 0.01s ie 10ms``

_shift_frames = model.encoder.streaming_cfg.shift_size
_shift_frames = _shift_frames[1] if isinstance(_shift_frames, list) else _shift_frames  # 32 (steady-state)

_precache_frames = model.encoder.streaming_cfg.pre_encode_cache_size
_precache_frames = _precache_frames[1] if isinstance(_precache_frames, list) else _precache_frames  # 9

CHUNK_SIZE_IN_SECS = _shift_frames * _window_stride          # 0.32s
BUFFER_SIZE_IN_SECS = (_shift_frames + _precache_frames) * _window_stride  # 0.41s
NEW_SAMPLES_PER_STEP = int(CHUNK_SIZE_IN_SECS * SAMPLE_RATE)  # 5120 raw samples

_drop_extra_pre_encoded = model.encoder.streaming_cfg.drop_extra_pre_encoded  # constant every step

# threading.Lock, not asyncio.Lock: the actual model call runs inside
# asyncio.to_thread worker threads, not on the event loop itself.
inference_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Feature extraction: ported from CacheFeatureBufferer (read from
# nemo/agents/voice_agent/pipecat/services/nemo/utils.py), instantiated
# per-session here since it holds mutable buffer state.
# ---------------------------------------------------------------------------
class _RawAudioRingBuffer:
    def __init__(self, sample_rate: int, buffer_size_in_secs: float):
        self.buffer_size = int(buffer_size_in_secs * sample_rate)
        self.samples = torch.zeros(self.buffer_size, dtype=torch.float32)

    def update(self, audio: np.ndarray):
        audio_t = torch.from_numpy(audio)
        shift = audio_t.shape[0]
        if shift > self.buffer_size:
            raise ValueError(f"chunk ({shift}) exceeds ring buffer size ({self.buffer_size})")
        self.samples[:-shift] = self.samples[shift:].clone()
        self.samples[-shift:] = audio_t.clone()


class CacheFeatureBufferer:
    """Ported from nemo/agents/voice_agent/pipecat/services/nemo/utils.py,
    scoped to CPU / this model's preprocessor config."""

    def __init__(self, preprocessor_cfg, device="cpu"):
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
            [self.n_feat, self.feature_buffer_len], self.zero_val, dtype=torch.float32, device=device
        )

        self.preprocessor = nemo_asr.models.ASRModel.from_config_dict(preprocessor_cfg)
        self.preprocessor.to(device)

    def _update_feature_buffer(self, feat_chunk: torch.Tensor):
        self.feature_buffer[:, : -self.feature_chunk_len] = self.feature_buffer[:, self.feature_chunk_len :].clone()
        self.feature_buffer[:, -self.feature_chunk_len :] = feat_chunk.clone()

    def _preprocess(self, audio_signal: torch.Tensor) -> torch.Tensor:
        audio_signal = audio_signal.unsqueeze(0).to(self.device)
        audio_signal_len = torch.tensor([audio_signal.shape[1]], device=self.device)
        features, _ = self.preprocessor(input_signal=audio_signal, length=audio_signal_len)
        return features.squeeze(0)

    def update(self, audio: np.ndarray):
        self.sample_buffer.update(audio)
        samples = self.sample_buffer.samples[-(self.n_chunk_look_back + self.chunk_size) :]
        features = self._preprocess(samples)
        if (diff := features.shape[1] - self.feature_chunk_len - 1) > 0:
            features = features[:, :-diff]
        self._update_feature_buffer(features[:, -self.feature_chunk_len :])

    def get_feature_buffer(self) -> torch.Tensor:
        return self.feature_buffer.clone()


def pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
    int16 = np.frombuffer(raw, dtype=np.int16)
    return int16.astype(np.float32) / 32768.0


class StreamingSession:
    def __init__(self, model):
        self.model = model
        self.feature_bufferer = CacheFeatureBufferer(model.cfg.preprocessor, device="cpu")

        # Raw-audio accumulator for whatever chunk sizes the client actually
        # sends over the socket -- we only run update()/conformer_stream_step
        # once NEW_SAMPLES_PER_STEP (320ms) of new audio has accumulated.
        self._pending = np.empty(0, dtype=np.float32)

        (
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        ) = model.encoder.get_initial_cache_state(batch_size=1)

        self.previous_hypotheses = None
        self.finalized = False
        self.transcript_text = ""

    def _run_one_step(self, new_audio: np.ndarray, keep_all_outputs: bool) -> str | None:
        self.feature_bufferer.update(new_audio)
        features = self.feature_bufferer.get_feature_buffer()  # (n_feat, feature_buffer_len)
        features = features.unsqueeze(0)  # (1, n_feat, T) -- batch dim
        feature_lengths = torch.tensor([features.shape[-1]])

        with inference_lock:
            with torch.inference_mode():
                t0 = time.perf_counter()
                (
                    _greedy_predictions,
                    _all_hyp,
                    self.cache_last_channel,
                    self.cache_last_time,
                    self.cache_last_channel_len,
                    best_hyp,
                ) = self.model.conformer_stream_step(
                    processed_signal=features,
                    processed_signal_length=feature_lengths,
                    cache_last_channel=self.cache_last_channel,
                    cache_last_time=self.cache_last_time,
                    cache_last_channel_len=self.cache_last_channel_len,
                    keep_all_outputs=keep_all_outputs,
                    previous_hypotheses=self.previous_hypotheses,
                    previous_pred_out=None,  # CTC-only field; unused for RNNT
                    drop_extra_pre_encoded=_drop_extra_pre_encoded,
                    return_transcription=True,
                    return_log_probs=False,
                )
                # CPU inference time
                infer_ms = (time.perf_counter() - t0) * 1000   
                print(f"Inference: {infer_ms:.1f} ms")

        self.previous_hypotheses = best_hyp
        energy = float(np.sqrt(np.mean(new_audio.astype(np.float32) ** 2)))
        if best_hyp is not None and len(best_hyp) > 0:
            self.transcript_text = best_hyp[0].text
            print(f"step energy: {energy:.5f} | text: {best_hyp[0].text!r} | y_sequence len: {len(best_hyp[0].y_sequence)}")
            return self.transcript_text
        print(f"step energy: {energy:.5f} | text: None | y_sequence len: 0")
        return None

    def process_chunk(self, raw_pcm_bytes: bytes) -> str | None:
        """Synchronous -- called via asyncio.to_thread from the WebSocket
        handler so blocking inference doesn't stall the event loop."""
        new_audio = pcm16_bytes_to_float32(raw_pcm_bytes)
        print(
            "samples:",
            len(new_audio),
            "min:",
            np.min(new_audio),
            "max:",
            np.max(new_audio),
            "mean abs:",
            np.mean(np.abs(new_audio)),
            "first 10:",
            new_audio[:10]
        )
        self._pending = np.concatenate([self._pending, new_audio])

        latest_text = None
        while len(self._pending) >= NEW_SAMPLES_PER_STEP:
            step_audio, self._pending = (
                self._pending[:NEW_SAMPLES_PER_STEP],
                self._pending[NEW_SAMPLES_PER_STEP:],
            )
            result = self._run_one_step(step_audio, keep_all_outputs=False)
            if result is not None:
                latest_text = result
        return latest_text

    def finalize(self) -> str:
        if self.finalized:
            return self.transcript_text
        self.finalized = True

        if len(self._pending) > 0:
            pad_len = NEW_SAMPLES_PER_STEP - len(self._pending)
            if pad_len > 0:
                self._pending = np.concatenate(
                    [self._pending, np.zeros(pad_len, dtype=np.float32)]
                )
            self._run_one_step(self._pending[:NEW_SAMPLES_PER_STEP], keep_all_outputs=False)
            self._pending = np.empty(0, dtype=np.float32)

        # Flush: RNNT streaming decoders often need a bit of trailing
        # silence to resolve the final pending symbol (the acoustic
        # evidence for the last phoneme partly lives just after where
        # real audio stopped). Feed a couple of pure-silence steps before
        # the true final step so nothing gets cut off mid-word.
        silence = np.zeros(NEW_SAMPLES_PER_STEP, dtype=np.float32)
        self._run_one_step(silence, keep_all_outputs=False)
        self._run_one_step(silence, keep_all_outputs=True)

        return self.transcript_text


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    session = StreamingSession(model)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                raw_pcm = message["bytes"]
                partial_text = await asyncio.to_thread(session.process_chunk, raw_pcm)
                if partial_text is not None:
                    await websocket.send_json({"type": "partial", "text": partial_text})

            elif "text" in message and message["text"] == "END":
                final_text = await asyncio.to_thread(session.finalize)
                await websocket.send_json({"type": "final", "text": final_text})
                await websocket.close()
                break

    except WebSocketDisconnect:
        try:
            session.finalize()
        except Exception:
            pass

# highest--->[startup] att_context_size=[56, 13] -> streaming_cfg=CacheAwareStreamingConfig(chunk_size=[105, 112], shift_size=[105, 112], cache_drop_size=0, last_channel_cache_size=56, valid_out_len=14, pre_encode_cache_size=[0, 9], drop_extra_pre_encoded=2, last_channel_num=0, last_time_num=0) -> chunk=1.12s buffer=1.21s (17920 raw samples/step, 1120ms per step)"""