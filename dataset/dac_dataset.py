import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL_MAP = {
    "backchannel": 0,
    "statement": 1,
    "question": 2,
    "opinion": 3
}


class DacDataset(Dataset):
    def __init__(self,
                 dataset_path: str,
                 audio_feat_dir: str,
                 tokenizer: str = "roberta-base"):
        self.audio_feat_dir = Path(audio_feat_dir)
        self.dataset_path = Path(dataset_path)

        self.max_len = None
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._load_audio_meta_data()
        self.max_len_text = 128

        self.documents = None
        self.class_weights = None
        self._load_documents()

    def _load_documents(self):
        documents = []
        label_counter = Counter()

        with open(self.dataset_path) as file_pointer:
            for index, line in enumerate(file_pointer):
                if line.strip():
                    id_, label, text = line.strip().split("\t")
                    documents.append({
                        "doc_id": id_,
                        "label": label,
                        "label_id": LABEL_MAP[label],
                        "text": text
                    })
                    label_counter.update([label])
        self.documents = documents

        document_number = len(label_counter)
        class_weights = [None] * len(LABEL_MAP)
        for label, index in LABEL_MAP.items():
            class_weights[index] = label_counter[label] / document_number
        self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def _load_audio_feat(self, doc_id: str) -> np.ndarray:
        audio_feat = np.load(str(Path(self.audio_feat_dir, f"{doc_id}.npy")))

        feat_len = audio_feat.shape[0]

        return np.pad(audio_feat, ((0, self.max_len - feat_len), (0, 0)))

    def _load_audio_meta_data(self):
        with open(Path(self.audio_feat_dir, "meta.json")) as file_pointer:
            meta = json.load(file_pointer)

        self.max_len = meta["max_len"]

    def _preprocess_text(self, text):
        tokens = self.tokenizer.tokenize(text)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_len_text - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == len(input_mask) == self.max_len_text

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(input_mask,
                                                                       dtype=torch.long)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, doc_id) -> T_co:
        doc = self.documents[doc_id]

        items = {}
        items['input_ids'], items['input_mask'] = self._preprocess_text(doc['text'])
        items['audio_feat'] = torch.tensor(self._load_audio_feat(doc["doc_id"]),
                                           dtype=torch.float)
        items['label'] = torch.tensor(doc['label_id'], dtype=torch.long)

        return items['label'], (items['input_ids'], items['input_mask']), items[
            'audio_feat']
