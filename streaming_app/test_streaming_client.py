"""
Minimal backend-validation client for the streaming ASR WebSocket server.

Streams a 16kHz mono WAV file as real-time 20ms PCM16 chunks (simulating a
live mic feed), sends "END" after the last chunk, and prints every partial
and final transcript received. Not for production use — just for validating
the FastAPI server before wiring up a real frontend.

Usage:
    pip install websockets
    python test_streaming_client.py path/to/audio.wav
"""

import asyncio
import sys
import time
import wave

import websockets

SERVER_URL = "ws://localhost:8000/ws/transcribe"
SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)  # 320 samples
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit PCM = 2 bytes/sample


def load_wav_chunks(path: str):
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz. "
                f"Resample first, e.g.:\n"
                f"  ffmpeg -i input.wav -ar {SAMPLE_RATE} -ac 1 -c:a pcm_s16le output.wav"
            )
        if wf.getnchannels() != 1:
            raise ValueError(f"Expected mono, got {wf.getnchannels()} channels.")
        if wf.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, got {wf.getsampwidth() * 8}-bit.")

        chunks = []
        while True:
            data = wf.readframes(CHUNK_SAMPLES)
            if not data:
                break
            chunks.append(data)
        return chunks


async def stream_audio(wav_path: str):
    chunks = load_wav_chunks(wav_path)
    print(f"Loaded {len(chunks)} chunks (~{len(chunks) * CHUNK_MS / 1000:.2f}s of audio)\n")

    async with websockets.connect(SERVER_URL) as ws:

        async def receiver():
            async for message in ws:
                # Server sends JSON: {"type": "partial"|"final", "text": "..."}
                import json
                msg = json.loads(message)
                if msg["type"] == "partial":
                    print(f"[partial] {msg['text']}")
                elif msg["type"] == "final":
                    print(f"\n[FINAL]   {msg['text']}")

        recv_task = asyncio.create_task(receiver())

        chunk_interval = CHUNK_MS / 1000.0
        next_send = time.monotonic()
        for chunk in chunks:
            await ws.send(chunk)
            next_send += chunk_interval
            sleep_for = next_send - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

        await ws.send("END")

        # Give the server a moment to send the final transcript, then close.
        try:
            await asyncio.wait_for(recv_task, timeout=10.0)
        except asyncio.TimeoutError:
            print("Timed out waiting for final transcript.")
            recv_task.cancel()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_streaming_client.py path/to/audio.wav")
        sys.exit(1)

    asyncio.run(stream_audio(sys.argv[1]))