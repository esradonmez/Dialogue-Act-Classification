import logging
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from acoustic_model.speech_cnn import SpeechCnn
from dataset.dac_dataset import DacDataset

DATA_PATH = "./data"
CACHE_PATH = "./cache"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s %(levelname)s %(name)s:%(lineno)s] %(message)s',
                              datefmt='%m/%d %H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

if __name__ == '__main__':
    logging.basicConfig(filename='speechCNN.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    learning_rate = 0.001
    batch_size = 8
    epochs = 10

    set_seed(42)

    file_handler = logging.FileHandler(os.path.join(CACHE_PATH, 'acoustic_5_13.txt'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpeechCnn(conv_kernel=(5, 13), pool_kernel=(5, 1))
    model.to(device)

    logger.info("Filter size: %s", (5, 13))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainset = DacDataset(
            f"{DATA_PATH}/train.txt",
            f"{DATA_PATH}/audio_features"
        )

    validset = DacDataset(
            f"{DATA_PATH}/dev.txt",
            f"{DATA_PATH}/audio_features"
        )
    trainloader = DataLoader(trainset, batch_size=batch_size)
    validloader = DataLoader(validset, batch_size=batch_size)

    for epoch in range(epochs):
        logger.info("Epoch: %5d", epoch)
        gold_labels = []
        pred_labels = []

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            labels, lexical_input, acoustic_input = data
            labels = labels.to(device)
            lexical_input = (lexical_input[0].to(device),lexical_input[1].to(device))
            acoustic_input = acoustic_input.to(device)

            gold_labels.extend(labels.cpu())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(acoustic_input)

            pred_labels.extend(torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())

            # backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    f"[{epoch + 1}\t{i+1}]\tloss: {running_loss / 2000}\t"
                    f"acc: {accuracy_score(gold_labels, pred_labels)}")
                running_loss = 0.0

        print(
            f"Finished epoch {epoch + 1}\t"
            f"acc: {accuracy_score(gold_labels, pred_labels)}")
        
        # Tracking variables
        total_eval_loss = 0
        gold_labels_val = []
        pred_labels_val = []

        for data in validloader:
            labels, lexical_input, acoustic_input = data
            labels = labels.to(device)
            lexical_input = (lexical_input[0].to(device),lexical_input[1].to(device))
            acoustic_input = acoustic_input.to(device)

            gold_labels_val.extend(labels.cpu())

            with torch.no_grad():
                outputs = model(acoustic_input)
                pred_labels_val.extend(torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())
                validloss = criterion(outputs, labels)
                total_eval_loss += validloss.item()

        avg_val_loss = total_eval_loss / len(validloader)
        #print("average_validation_loss:", avg_val_loss)

        print(
            f"Finished epoch {epoch + 1}\n"
            f"Training accuracy: {accuracy_score(gold_labels, pred_labels)}\n"
            f"Validation accuracy: {accuracy_score(gold_labels_val, pred_labels_val)}")
            
        logger.info("Average validation loss: %.5f", avg_val_loss)
        logger.info("Training accuracy: %.2f", accuracy_score(gold_labels, pred_labels))
        logger.info("Validation accuracy: %.2f", accuracy_score(gold_labels_val, pred_labels_val))
