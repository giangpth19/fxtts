import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
import io
import soundfile as sf

chunks = [
    "hôm nay tôi muốn đi siêu thị",
    "nhưng trời mưa rất to",
    "có thể tôi sẽ ở nhà",
    "và đặt đồ ăn qua mạng"
]

sent = " ".join(chunks)

# payload_vi = {
#     "lang": "vi",
#     "text": "Bạn có thể giải thích giúp tôi một cách rõ ràng về vấn đề đang gặp phải không? Nếu việc này tốn thời gian của bạn, chúng ta có thể để đến cuộc họp chính thức.",
#     "speaker_wav": "ref/vi_male.wav"
# }

vi_text = "xin chào. hôm nay tôi muốn đi công viên. màu trời rất đẹp"

payload_vi = {
    "lang": "vi",
    "text": vi_text,
    "speaker_wav": "ref/vi.wav"
}

payload_en = {
    "lang": "en",
    "text":"Can you explain clearly what the problem is? Will this take up your time? If so, we can move on to a formal meeting.",
    "speaker_wav": "ref/en.wav"
}


payload_ja = {
    "lang": "ja",
    "text":"何が問題なのか、わかりやすく説明していただけますか？お時間かかりますでしょうか？もしそうであれば、正式な打ち合わせに移りましょう.",
    "speaker_wav": "ref/ja.wav"
}

SERVER_URL = "ws://localhost:8765"

async def tts_request(text, lang="vi", speaker_wav=None):
    async with websockets.connect(SERVER_URL, ping_timeout=None) as websocket:
        # Prepare and send the message
        request_data = {
            "text": text,
            "lang": lang,
            "speaker_wav": speaker_wav  # Can be None if server handles default
        }
        await websocket.send(json.dumps(request_data))
        print(f"[CLIENT] Sent request: {request_data}")

        # Wait for response
        response = await websocket.recv()
        if isinstance(response, bytes):
            print("[CLIENT] Received audio bytes")

            # Convert bytes to numpy array (float32)
            audio_bytes = io.BytesIO(response)
            audio_np, samplerate = sf.read(audio_bytes, dtype='float32')

            print(f"[CLIENT] Playing audio... (Sample rate: {samplerate}, Shape: {audio_np.shape})")
            sd.play(audio_np, samplerate=samplerate)
            sd.wait()
            sf.write(str("output_ja.wav"), audio_np.squeeze(), samplerate=16000)

        else:
            try:
                err = json.loads(response)
                print(f"[CLIENT] Error response: {err}")
            except Exception:
                print("[CLIENT] Received non-audio, non-JSON response")

if __name__ == "__main__":
    text = "Xin chào, đây là một thử nghiệm chuyển văn bản thành giọng nói."
    lang = "vi"  # or "en", "ja"
    speaker_wav = "ref/vi.wav"  # Can use "ref/vi.wav" or leave as None
    payload = payload_vi

    asyncio.run(tts_request(text=payload['text'], lang=payload['lang'], speaker_wav=payload['speaker_wav']))
 