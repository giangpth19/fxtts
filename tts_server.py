import asyncio
import websockets
import json
import io
import torch
import numpy as np
import soundfile as sf
from tts_module import TTSModule

# Load all models once
tts = TTSModule()

# Helper to convert tensor to .wav bytes
def tensor_to_wav_bytes(wav_tensor, sample_rate=16000):
    buffer = io.BytesIO()
    sf.write(buffer, wav_tensor, samplerate=sample_rate, format='WAV', subtype='PCM_16')
    buffer.seek(0)
    return buffer.read()

# WebSocket handler
async def tts_websocket(websocket):
    async for message in websocket:
        try:
            data = json.loads(message)
            lang = data["lang"]
            text = data["text"]
            speaker_wav = data["speaker_wav"]

            print("[INFO] Received payload: lang={}, text={}, speaker_wav={}".format(lang, text, speaker_wav))

            model = tts._get_model(lang)

            wav = model.synthesize(text, lang=lang, speaker_wav=speaker_wav)

            # Convert to .wav bytes
            wav_bytes = tensor_to_wav_bytes(wav)
            await websocket.send(wav_bytes)

        except Exception as e:
            error_msg = f"[ERROR] {str(e)}"
            await websocket.send(json.dumps({"error": error_msg}))
            print(error_msg)

# Start server
if __name__ == "__main__":
    async def main():
        print("Starting TTS WebSocket server at ws://localhost:8765")
        async with websockets.serve(tts_websocket, "localhost", 8765, ping_timeout=None):
            await asyncio.Future()  # run forever

    asyncio.run(main())