import torch
import torch.nn as nn
import random
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from acoustic_model import SpeechCnn
from combined_model import CombinedModel
from dataset import DacDataset
from language_model import ContextAwareDAC

DATA_PATH = "./data"
CACHE_PATH = "./cache/dac"

if __name__ == '__main__':
    learning_rate = 0.001
    batch_size = 32
    epochs = 10
    
    torch.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CombinedModel(
        acoustic_model=SpeechCnn(),
        lexical_model=ContextAwareDAC(device=device)
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainset = DacDataset(
        f"{DATA_PATH}/train.txt",
        f"{DATA_PATH}/audio_features"
    )

    validset = DacDataset(
        f"{DATA_PATH}/dev.txt",
        f"{DATA_PATH}/audio_features",
        "roberta-base"
    )
    trainloader = DataLoader(trainset, batch_size=batch_size)
    validloader = DataLoader(validset, batch_size=batch_size)
        
    for epoch in range(epochs):
        gold_labels = []
        pred_labels = []

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):

            labels, lexical_input, acoustic_input = data
            labels = labels.to(device)
            lexical_input = (lexical_input[0].to(device),lexical_input[1].to(device))
            acoustic_input = acoustic_input.to(device)

            gold_labels.extend(labels.cpu())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(lexical_input, acoustic_input)

            pred_labels.extend(torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())

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

        print("\nRunning Validation...")

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
                outputs = model(lexical_input, acoustic_input)
                pred_labels_val.extend(torch.argmax(torch.softmax(outputs, 1), 1).detach().cpu().numpy())
                validloss = criterion(outputs, labels)
                total_eval_loss += validloss.item()

        avg_val_loss = total_eval_loss / len(validloader)
        print("average_validation_loss:", avg_val_loss)

        print(
            f"Finished epoch {epoch + 1}\n"
            f"Training accuracy: {accuracy_score(gold_labels, pred_labels)}\n"
            f"Validation accuracy: {accuracy_score(gold_labels_val, pred_labels_val)}")

        # save the model
        torch.save(model.state_dict(), CACHE_PATH+str(epoch+1)+".ckpt")