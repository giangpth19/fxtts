# tts/backends/vixtts_backend.py
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
from underthesea import sent_tokenize
from utils import normalize_vietnamese_text, calculate_keep_len
import soundfile as sf
from .base import BaseTTSBackend

class ViXTTSBackend(BaseTTSBackend):
    def __init__(self, model_path, device):
        config = XttsConfig()
        config.load_json(str(model_path / "config.json"))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=str(model_path), use_deepspeed=False)
        if device == "cuda":
            model.cuda()
        model.eval()

        self.model = model
        self.config = config
        self.device = device

    def synthesize(self, text, lang=None, speaker_wav=None):
        assert speaker_wav, "viXTTS requires speaker_wav"

        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_wav,
            gpt_cond_len=self.config.gpt_cond_len,
            max_ref_length=self.config.max_ref_len,
            sound_norm_refs=self.config.sound_norm_refs,
        )

        text = normalize_vietnamese_text(text)
        tts_texts = sent_tokenize(text)
        wav_chunks = []

        for sent in tts_texts:
            if not sent.strip():
                continue
            result = self.model.inference(
                text=sent,
                language=lang,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.3,
                length_penalty=1.0,
                repetition_penalty=10.0,
                top_k=30,
                top_p=0.85,
            )

            # Quick hack for short sentences
            keep_len = calculate_keep_len(text, lang)
            result["wav"] = torch.tensor(result["wav"][:keep_len])
            wav_chunk = torch.tensor(result["wav"], dtype=torch.float32)

            if wav_chunk.ndim == 1:
                wav_chunk = wav_chunk.unsqueeze(1)
            wav_chunks.append(wav_chunk)

        return torch.cat(wav_chunks, dim=0)

    def save_wav(self, wav, path):
        sf.write(str(path), wav.squeeze(), samplerate=24000)
 