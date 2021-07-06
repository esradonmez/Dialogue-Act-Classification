import random
import logging

import torch
import numpy as np
from pathlib import Path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_logger(name: str, file_path: Path) -> logging.Logger:
    file_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s %(levelname)s %(name)s:%(lineno)s] %(message)s',
        datefmt='%m/%d %H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger
