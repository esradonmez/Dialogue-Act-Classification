import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from acoustic_model.speech_cnn import SpeechCnn
from acoustic_model.dac_dataset import DacDataset

DATA_PATH = "./data"

if __name__ == '__main__':
    model = SpeechCnn()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset = DacDataset(
        f"{DATA_PATH}/train.txt",
        f"{DATA_PATH}/audio_features"
    )
    dataloader = DataLoader(dataset, batch_size=8)

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            labels, inputs = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            # print(f"{outputs=}")
            loss = criterion(outputs, labels)
            # print(f"{loss=}")
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0