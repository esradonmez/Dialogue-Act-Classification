import json
from pathlib import Path

import numpy as np
from torch.utils.data.dataset import T_co
from torch.utils.data import IterableDataset


LABEL_MAP = {
    "backchannel": 0,
    "statement": 1,
    "question": 2,
    "opinion": 3
}


class DacDataset(IterableDataset):
    def __init__(self, dataset_path: str, audio_feat_dir: str):
        self.audio_feat_dir = Path(audio_feat_dir)
        self.dataset_path = Path(dataset_path)

        self.max_len = None
        self._load_audio_meta_data()

        self.documents = self._load_documents()

    def _load_documents(self):
        documents = {}

        with open(self.dataset_path) as file_pointer:
            for line in file_pointer:
                if line.strip():
                    id_, label, text = line.strip().split("\t")
                    documents[id_] = {
                        "label": label,
                        "label_id": LABEL_MAP[label],
                        "text": text
                    }
        return documents

    def _load_audio_feat(self, doc_id: str) -> np.ndarray:
        with open(Path(self.audio_feat_dir, f"{doc_id}.npy"), "rb") as file_pointer:
            audio_feat = np.load(file_pointer)

        feat_len = audio_feat.shape[0]

        return np.pad(audio_feat, ((0, self.max_len - feat_len), (0, 0)))

    def _load_audio_meta_data(self):
        with open(Path(self.audio_feat_dir, "meta.json")) as file_pointer:
            meta = json.load(file_pointer)

        self.max_len = meta["max_len"]

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, doc_id) -> T_co:
        return self.documents[doc_id]["label_id"], self._load_audio_feat(doc_id)

    def __iter__(self):
        for doc_id, doc in self.documents.items():
            yield doc["label_id"], self._load_audio_feat(doc_id)
