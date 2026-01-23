import os

import numpy as np
import soundfile as sf


def create_dummy_wav(filename="dummy_audio.wav", duration=5.0, sr=16000):
    """Creates a dummy sine wave file (WAV)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    sf.write(filename, x, sr)

if __name__ == "__main__":
    create_dummy_wav("tests/performancetests/dummy_audio.wav")
