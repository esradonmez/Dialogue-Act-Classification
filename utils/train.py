import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import DacDataset
from utils import setup_logger, set_seed, weights_init

load_dotenv(".env")
LOG_PATH = Path(os.getenv("SDS_DAC_LOG_PATH", "./logs"))
DATA_PATH = os.getenv("SDS_DAC_DATA_PATH", "./data")
CACHE_PATH = Path(os.getenv("SDS_DAC_CACHE_PATH", "./cache"))
DEVICE = torch.device(
    os.getenv("SDS_DAC_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))


def format_parameters(**kwargs) -> str:
    return "-".join(f"{k}={v}" for k, v in kwargs.items())


def train(
        model: nn.Module,
        model_type: str = "combined",
        batch_size: int = 16,
        epochs: int = 10,
        learning_rate: float = 0.001,
        balance_loss: bool = False,
        balance_data: bool = False
):
    assert not (balance_loss and balance_data)
    param_format = format_parameters(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        balance_loss=balance_loss,
        balance_data=balance_data
    )
    logger = setup_logger(__name__, Path(LOG_PATH, model_type, f"{param_format}.log"))
    set_seed(42)
    model.to(DEVICE)

    # init dataloaders
    train_set = DacDataset(
        f"{DATA_PATH}/train.txt",
        f"{DATA_PATH}/audio_features"
    )
    train_sample_weight = [train_set.class_weights[doc["label_id"]]
                           for doc in train_set.documents]
    train_sampler = WeightedRandomSampler(train_sample_weight, len(train_sample_weight))
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler if balance_data else None)

    val_set = DacDataset(
        f"{DATA_PATH}/dev.txt",
        f"{DATA_PATH}/audio_features",
        "roberta-base"
    )
    val_loader = DataLoader(val_set, batch_size=batch_size)

    class_weights = train_set.class_weights.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights if balance_loss else None)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.apply(weights_init)

    # parameters for early stopping
    es_min_val_loss = 10000
    es_counter = 0
    es_patience = 6  # default: wait for 6 epochs
    es_min_epochs = 5

    for epoch in range(epochs):
        gold_labels = []
        pred_labels = []

        for i, train_data in enumerate(train_loader, 0):
            labels, lexical_input, acoustic_input = train_data
            labels = labels.to(DEVICE)
            lexical_input = (lexical_input[0].to(DEVICE), lexical_input[1].to(DEVICE))
            acoustic_input = acoustic_input.to(DEVICE)

            gold_labels.extend(labels.cpu())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            if model_type == "acoustic":
                outputs = model(acoustic_input)
            elif model_type == "lexical":
                outputs = model(input_ids=lexical_input[0],
                                attention_mask=lexical_input[1])
            elif model_type == "combined":
                outputs = model(lexical_input, acoustic_input)
            else:
                raise Exception("Shit`s fucked yo!")

            pred_labels.extend(
                torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())

            # backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Tracking variables
        total_val_loss = 0
        val_gold_labels = []
        val_pred_labels = []
        # validate
        for val_data in val_loader:
            val_labels, val_lexical_input, val_acoustic_input = val_data
            val_labels = val_labels.to(DEVICE)
            val_lexical_input = (
                val_lexical_input[0].to(DEVICE),
                val_lexical_input[1].to(DEVICE))
            val_acoustic_input = val_acoustic_input.to(DEVICE)

            val_gold_labels.extend(val_labels.cpu())

            with torch.no_grad():
                # forward
                if model_type == "acoustic":
                    val_outputs = model(val_acoustic_input)
                elif model_type == "lexical":
                    val_outputs = model(*val_lexical_input)
                elif model_type == "combined":
                    val_outputs = model(val_lexical_input, val_acoustic_input)
                else:
                    raise Exception("Shit`s fucked yo!")

                val_pred_labels.extend(
                    torch.argmax(
                        torch.softmax(val_outputs, 1), 1).detach().cpu().numpy())
                val_loss = criterion(val_outputs, val_labels)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logger.info("Epoch: %5d", epoch + 1)
        logger.info("Average validation loss: %.5f", avg_val_loss)
        logger.info("Training accuracy: %.4f", accuracy_score(gold_labels, pred_labels))
        logger.info("Validation accuracy: %.4f",
                    accuracy_score(val_gold_labels, val_pred_labels))

        # save the model
        torch.save(model.state_dict(),
                   Path(CACHE_PATH,
                        model_type, f"{param_format}-epoch={epoch + 1}.ckpt"))

        # early stopping
        # wait 3 epochs before recording loss to allow model to warm up
        if epoch > 3 and avg_val_loss < es_min_val_loss:
            es_min_val_loss = avg_val_loss
            es_counter = 0
        else:
            es_counter += 1
        # do stop if we are out of the min epoch range and lost patience
        if epoch > es_min_epochs and es_counter >= es_patience:
            logger.info(f'Early stopping at epoch {epoch}!')
            break
