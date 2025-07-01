# tts/backends/base.py
from abc import ABC, abstractmethod
import numpy as np 

class BaseTTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text, lang=None, speaker_wav=None):
        pass

    @abstractmethod
    def save_wav(self, wav, path):
        pass