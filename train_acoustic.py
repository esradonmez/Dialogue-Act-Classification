import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from acoustic_model.speech_cnn import SpeechCnn
from dataset.dac_dataset import DacDataset

DATA_PATH = "./data"

if __name__ == '__main__':
    learning_rate = 0.001
    batch_size = 8
    epochs = 5

    model = SpeechCnn()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    dataset = DacDataset(
        f"{DATA_PATH}/train.txt",
        f"{DATA_PATH}/audio_features"
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for epoch in range(epochs):
        gold_labels = []
        pred_labels = []

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):

            labels, inputs = data
            gold_labels.extend(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs.float())

            pred_labels.extend(torch.argmax(torch.softmax(outputs, 1), 1))

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
