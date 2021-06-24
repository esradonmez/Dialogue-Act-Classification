import argparse
import os
import random
import sys
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
CACHE_PATH = "./test/dac.ckpt"


def run(gpu_id, model_dir):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(13)
    random.seed(13)

    learning_rate = 0.001
    batch_size = 32
    epochs = 5

    model = CombinedModel(
        acoustic_model=SpeechCnn(),
        lexical_model=ContextAwareDAC(device=device),
        lexical_tokenizer=RobertaTokenizer.from_pretrained("roberta-base")
    )

    model = model.to(device)
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

    print("Training model")
    model.train()

    for epoch in range(epochs):
        gold_labels = []
        pred_labels = []

        running_loss = 0.0
        for i, data in enumerate(tqdm(trainloader), 0):

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


        print("")
        print("Running Validation...")
        model.eval()

        # Tracking variables
        total_eval_loss = 0
        gold_labels_val = []
        pred_labels_val = []

        for labels, combined_inputs in validloader:
            gold_labels_val.extend(labels)

            with torch.no_grad():
                outputs = model(combined_inputs)
                pred_labels_val.extend(torch.argmax(torch.softmax(outputs, 1), 1))
                validloss = criterion(outputs, labels)
                total_eval_loss += validloss.item()

        avg_val_loss = total_eval_loss / len(validloader)
        print("average_validation_loss:", avg_val_loss)

        print(
            f"Finished epoch {epoch + 1}\t"
            f"acc: {accuracy_score(gold_labels, pred_labels)}"
            f"acc: {accuracy_score(gold_labels_val, pred_labels_val)}")

        # save the model
        torch.save(model.state_dict(), CACHE_PATH + str(epoch + 1))

if __name__ == '__main__':

    torch.multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='DAC project')

    parser.add_argument('--gpu_id',
                        type=str,
                        help="Which GPU to run on. If not specified runs on CPU, but other than for integration tests that doesn't make much sense.",
                        default="cpu")

    parser.add_argument('--model_save_dir',
                        type=str,
                        help="Directory where the checkpoints should be saved to.",
                        default=CACHE_PATH)

    args = parser.parse_args()

    if args.finetune and args.resume_checkpoint is None:
        print("Need to provide path to checkpoint to fine-tune from!")
        sys.exit()

    run(gpu_id=args.gpu_id,model_dir=args.model_save_dir)
