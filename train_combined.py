import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from tqdm import tqdm

from acoustic_model import SpeechCnn
from combined_model import CombinedModel
from dataset import DacDataset
from language_model import ContextAwareDAC

DATA_PATH = "./data"

if __name__ == '__main__':
    learning_rate = 0.001
    batch_size = 32
    epochs = 5

    model = CombinedModel(
        acoustic_model=SpeechCnn(),
        lexical_model=ContextAwareDAC(device="cpu"),
        lexical_tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
    )

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
        for i, data in enumerate(tqdm(dataloader), 0):

            labels, combined_inputs = data
            gold_labels.extend(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(combined_inputs)

            pred_labels.extend(torch.argmax(torch.softmax(outputs, 1), 1))

            # backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    f"[{epoch + 1}\t{i + 1}]\tloss: {running_loss / 2000}\t"
                    f"acc: {accuracy_score(gold_labels, pred_labels)}")
                running_loss = 0.0

        print(
            f"Finished epoch {epoch + 1}\t"
            f"acc: {accuracy_score(gold_labels, pred_labels)}")
