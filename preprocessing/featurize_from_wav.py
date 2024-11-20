import numpy as np
import pandas as pd
import os, glob
import librosa

def featurize_from_wav(wav_path):
    sample_rate = 48000
    