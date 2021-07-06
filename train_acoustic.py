import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from dataset import DacDataset
from models import SpeechCnn
from utils import setup_logger, set_seed

load_dotenv(".env")
TRAIN_TAG = "acoustic"
DATA_PATH = os.getenv("SDS_DAC_DATA_PATH", "./data")
CACHE_PATH = Path(os.getenv('SDS_DAC_CACHE_PATH', './cache'), TRAIN_TAG)
LOG_PATH = Path(os.getenv('SDS_DAC_LOG_PATH', './logs'), TRAIN_TAG)

CACHE_PATH.mkdir(parents=True, exist_ok=True)


def train():
    logger = setup_logger(__name__, Path(LOG_PATH, "train.log"))

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model parameters
    conv_kernel = (10, 13)
    pool_kernel = (10, 1)
    pool_stride = 6
    model = SpeechCnn(
        conv_kernel=conv_kernel,
        pool_kernel=pool_kernel,
        pool_stride=pool_stride)
    model.to(device)
    logger.info("Filter size: %s", conv_kernel)

    # define training parameters
    learning_rate = 0.0001
    batch_size = 16
    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # init dataloaders
    train_set = DacDataset(
        f"{DATA_PATH}/train.txt",
        f"{DATA_PATH}/audio_features"
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    valid_set = DacDataset(
        f"{DATA_PATH}/dev.txt",
        f"{DATA_PATH}/audio_features"
    )
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    for epoch in range(epochs):
        gold_labels = []
        pred_labels = []

        for i, data in enumerate(train_loader, 0):
            labels, _, acoustic_input = data
            labels = labels.to(device)
            acoustic_input = acoustic_input.to(device)

            gold_labels.extend(labels.cpu())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(acoustic_input)

            pred_labels.extend(
                torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())

            # backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Tracking variables
        total_eval_loss = 0
        gold_labels_val = []
        pred_labels_val = []

        for data in valid_loader:
            labels, _, acoustic_input = data
            labels = labels.to(device)
            acoustic_input = acoustic_input.to(device)

            gold_labels_val.extend(labels.cpu())

            with torch.no_grad():
                outputs = model(acoustic_input)
                pred_labels_val.extend(
                    torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())
                validloss = criterion(outputs, labels)
                total_eval_loss += validloss.item()

        avg_val_loss = total_eval_loss / len(valid_loader)
        logger.info("Epoch: %5d", epoch + 1)
        logger.info("Average validation loss: %.5f", avg_val_loss)
        logger.info("Training accuracy: %.4f", accuracy_score(gold_labels, pred_labels))
        logger.info("Validation accuracy: %.4f",
                    accuracy_score(gold_labels_val, pred_labels_val))

        # save the model
        torch.save(model.state_dict(), Path(CACHE_PATH, f"{epoch + 1}.ckpt"))


if __name__ == '__main__':
    train()
