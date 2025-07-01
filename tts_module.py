import os 
from pathlib import Path
import torch
from huggingface_hub import snapshot_download, hf_hub_download
import numpy as np
import soundfile as sf
from scipy.signal import resample

from backends.xtts_backend import XTTSBackend
from backends.vixtts_backend import ViXTTSBackend

class TTSWrapper:
    def __init__(self, model_id, model_path, device="auto"):
        self.backend = self._load_backend(model_id, model_path, device)

    def _load_backend(self, model_id, model_path, device):
        if "capleaf/viXTTS" in model_id:
            return ViXTTSBackend(model_path, device)
        else:
            return XTTSBackend(model_path, device)

    def synthesize(self, text, lang=None, speaker_wav=None):
        wav = self.backend.synthesize(text, lang, speaker_wav)

        # XTTS returns torch tensor
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()

        # viXTTS already returns np array (after torch.cat + save)

        # Always convert to float32 and resample to 16kHz
        wav = wav.astype(np.float32)
        wav = resample_audio(wav, orig_sr=24000, target_sr=16000)
        return wav    

    def batch_synthesize(self, text, lang=None, speaker_wav=None):
        return self.backend.batch_synthesize(text, lang, speaker_wav)

    def save_wav(self, wav, path):
        # self.backend.save_wav(wav, path)
        sf.write(str(path), wav.squeeze(), samplerate=24000)

class TTSModule:
    def __init__(self, lang_to_model=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if lang_to_model: 
            self.lang_to_model = lang_to_model
        else: 
            self.lang_to_model = {
                "en": "coqui/XTTS-v2",
                "ja": "coqui/XTTS-v2",
                "vi": "capleaf/viXTTS"
            }

        self.base_dir = Path("tts_models")
        self.models = {}
        self.model_paths = self._download_models()
        self._load_all_models()

    def _download_models(self):
        model_paths = {}
        for model_id in set(self.lang_to_model.values()):
            local_dir = self.base_dir / model_id.replace("/", os.sep)
            os.makedirs(local_dir, exist_ok=True)

            # Check if model files already exist
            required_files = ["config.json", "model.pth", "vocab.json", "speakers_xtts.pth"]
            if all((local_dir / f).exists() for f in required_files):
                print(f"[INFO] Using cached model: {model_id}")
                model_paths[model_id] = local_dir
            else:
                print(f"[INFO] Downloading model: {model_id}")
                path = snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                model_paths[model_id] = Path(path)

                # viXTTS extra file
                # if "viXTTS" in model_id:
                #     print("[INFO] Downloading speakers_xtts.pth from coqui/XTTS-v2")
                #     hf_hub_download(
                #         repo_id="coqui/XTTS-v2",
                #         filename="speakers_xtts.pth",
                #         local_dir=local_dir,
                #     )

        return model_paths

    def _load_all_models(self):
        for lang, model_id in self.lang_to_model.items():
            model_path = self.model_paths[model_id]
            self.models[lang] = TTSWrapper(model_id, model_path, self.device)

    def _get_model(self, lang):
        return self.models[lang]

    def speak(self, lang, text, speaker_wav=None, filename="output.wav"):
        model = self._get_model(lang)
        wav = model.synthesize(text, lang=lang, speaker_wav=speaker_wav)
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save_wav(wav, path)
        print(f"[INFO] Saved: {filename}")

    def speak_batch(self, lang, texts, speaker_wav=None, filename="batch.wav"):
        model = self._get_model(lang)

        sample_rate = 24000
        silence = np.zeros(int(sample_rate * 0.3))

        wavs = [model.synthesize(t, lang=lang, speaker_wav=speaker_wav) for t in texts]
        full_wav = np.concatenate([np.concatenate([w, silence]) for w in wavs])
        model.save_wav(full_wav, filename)
        print(f"[INFO] Saved: {filename}")


def resample_audio(wav, orig_sr, target_sr=16000):
    if orig_sr == target_sr:
        return wav
    num_samples = int(len(wav) * target_sr / orig_sr)
    return resample(wav, num_samples)