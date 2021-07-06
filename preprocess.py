from pathlib import Path

from dataset import prepare_folder, generate_mfcc_features

DATA_PATH = "./data"

if __name__ == '__main__':
    prepare_folder(Path(DATA_PATH, "audio_features"))

    generate_mfcc_features(
        Path(DATA_PATH, "audios"),
        Path(DATA_PATH, "audio_features"))
