# tts/backends/xtts_backend.py
from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

from .base import BaseTTSBackend

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])

class XTTSBackend(BaseTTSBackend):
    def __init__(self, model_path, device):
        self.model = TTS(model_path=model_path, config_path=model_path / "config.json")
        self.model.to(device)

    def synthesize(self, text, lang=None, speaker_wav=None):
        assert lang and speaker_wav, "XTTS requires language and speaker_wav"
        wav = self.model.tts(text, speaker_wav=speaker_wav, language=lang)
        # Convert to torch.Tensor and ensure shape (T, 1)
        wav = torch.tensor(wav, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(1)
        return wav

    def save_wav(self, wav, path):
        self.model.synthesizer.save_wav(wav, str(path))
 