import os
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tqdm import tqdm

DATA_PATH = "../data"


def prepare_folder(folder_path: Path):
    if not folder_path.exists():
        os.mkdir(folder_path)


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

        with open(Path(output_dir, f"{audio_file.stem}.npy"), "wb") as file_pointer:
            np.save(file_pointer, mfcc_feat)

    with open(Path(output_dir, "meta.json"), "w") as file_pointer:
        json.dump({
            "max_len": max_len
        }, file_pointer)


if __name__ == '__main__':
    prepare_folder(Path(DATA_PATH, "audio_features"))

    generate_mfcc_features(
        Path(DATA_PATH, "audios"),
        Path(DATA_PATH, "audio_features"))
