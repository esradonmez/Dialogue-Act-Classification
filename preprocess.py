from pathlib import Path

from dataset import generate_mfcc_features

DATA_PATH = "./data"

if __name__ == '__main__':
    Path(DATA_PATH, "audio_features").mkdir(parents=True, exist_ok=True)

    generate_mfcc_features(
        Path(DATA_PATH, "audios"),
        Path(DATA_PATH, "audio_features"))
