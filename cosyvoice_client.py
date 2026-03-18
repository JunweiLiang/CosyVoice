# coding=utf-8
import argparse
import requests
import torch
import torchaudio.functional as F
import numpy as np
import time
import sounddevice as sd
import threading
import queue
import sys

# --- 1. Device Setup ---
devices = sd.query_devices()
if len(devices) == 0:
    print("No audio devices found", file=sys.stderr)
    sys.exit(0)

# Force the USB Camera for both input (8) and output (8)
sd.default.device = (8, 8)
input_device_idx, output_device_idx = sd.default.device
print(f"Default is using this speaker: {devices[output_device_idx]['name']}")

# --- 2. Real-Time Streaming Variables ---
audio_queue = queue.Queue()
# Buffer initialized as int16 to match standard hardware formats
leftover_chunk = np.zeros((0, 1), dtype=np.int16)
stop_playback_event = threading.Event()

def _audio_playback_callback(outdata, frames, time_info, status):
    """
    Consumer: Pulls audio chunks from the queue and feeds them to the speaker in real-time.
    Runs in an internal sounddevice C-thread.
    """
    global leftover_chunk, stop_playback_event

    if status and status.output_underflow:
        print('Warning: Network downloading slower than playback rate.', file=sys.stderr)

    current_idx = 0
    while current_idx < frames:
        # If we ran out of leftover audio from the last chunk, grab a new one
        if len(leftover_chunk) == 0:
            try:
                leftover_chunk = audio_queue.get_nowait()
                # If we pull 'None', the download loop has explicitly told us it's finished
                if leftover_chunk is None:
                    stop_playback_event.set()
                    outdata[current_idx:].fill(0)
                    raise sd.CallbackStop
            except queue.Empty:
                # Buffer underrun (download is lagging behind playback).
                # Fill the rest of the current frame with silence.
                outdata[current_idx:].fill(0)
                return

        # Figure out how much of the hardware frame we can fill with our current chunk
        take = min(frames - current_idx, len(leftover_chunk))
        outdata[current_idx:current_idx + take] = leftover_chunk[:take]

        # Advance the pointers
        current_idx += take
        leftover_chunk = leftover_chunk[take:]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default='50000')
    parser.add_argument('--tts_text', type=str,
                        default='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。')
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/inference_zero_shot_fast"
    orig_sr = 24000
    target_sr = 48000 # The specific sample rate required by the USB Camera ALSA driver

    payload = {'tts_text': args.tts_text}

    # Initialize the audio stream, strictly enforcing int16 dtype
    audio_stream = sd.OutputStream(
        samplerate=target_sr,
        channels=1,
        dtype=np.int16,
        callback=_audio_playback_callback,
    )

    stop_playback_event.clear()

    print("Sending request to CosyVoice API...")
    start_time = time.perf_counter()
    response = requests.request("GET", url, data=payload, stream=True)

    # Start the hardware speaker IMMEDIATELY before we even process the first chunk
    audio_stream.start()
    first_chunk_played = False

    # --- 3. The Producer Loop ---
    # We grab 8000 bytes at a time (equivalent to ~0.16 seconds of 24kHz audio)
    for r in response.iter_content(chunk_size=8000):
        if not r:
            continue

        if not first_chunk_played:
            print("Took %.3f seconds to first voice!" % (time.perf_counter() - start_time))
            first_chunk_played = True

        # 1. Convert raw bytes straight into an int16 numpy array
        raw_speech = np.frombuffer(r, dtype=np.int16)

        # 2. Temporarily cast to float32 ONLY because PyTorch's resample requires it
        audio_tensor = torch.from_numpy(raw_speech).to(torch.float32)

        # 3. Resample the chunk from 24kHz to 48kHz
        resampled_tensor = F.resample(audio_tensor, orig_freq=orig_sr, new_freq=target_sr)

        # 4. Cast straight back to int16 (No normalization math required)
        chunk_np = resampled_tensor.to(torch.int16).numpy().reshape(-1, 1)

        # 5. Push the processed chunk to the thread-safe queue for the callback to play
        audio_queue.put(chunk_np)

    # Signal the callback that the download is completely finished
    audio_queue.put(None)

    # Keep the main Python thread alive until the C-callback finishes playing the very last chunk
    while True:
        if not audio_stream.active or stop_playback_event.is_set():
            break
        time.sleep(0.05)

    audio_stream.close()
    print("Playback complete.")


