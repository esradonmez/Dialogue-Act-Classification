import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tqdm import tqdm


def generate_mfcc_features(input_dir: Path, output_dir: Path):
    input_size = len(list(input_dir.iterdir()))
    max_len = -1

    for audio_file in tqdm(input_dir.iterdir(), total=input_size):
        if audio_file.suffix != ".wav":
            continue

        rate, sig = wav.read(audio_file)
        mfcc_feat = mfcc(sig, rate)

        if mfcc_feat.shape[0] > max_len:
            max_len = mfcc_feat.shape[0]

        np.save(f"{audio_file.stem}.npy", mfcc_feat)

    with open(Path(output_dir, "meta.json"), "w") as file_pointer:
        json.dump({
            "max_len": max_len
        }, file_pointer)
