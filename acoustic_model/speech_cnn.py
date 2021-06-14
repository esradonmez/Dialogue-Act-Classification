import torch
import torch.nn.functional as F
from torch import nn


def conv_output_dim(input_dim, kernel, stride, padding=0):
    return int(((input_dim + 2 * padding - (kernel - 1) - 1) / stride) + 1)


def pool_output_dim(input_dim, kernel, stride, padding=0):
    return int(((input_dim - kernel + 2 * padding) / stride) + 1)


class SpeechCnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.output_pool_width = pool_output_dim(
            conv_output_dim(13, 5, 1, 0), 5, 3, 0)

        self.output_pool_height = pool_output_dim(
            conv_output_dim(3361, 5, 1, 0), 5, 3, 0)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(5, 5)
        )  # i'm not sure about these

        self.pool = nn.MaxPool2d(
            kernel_size=(5, 5),
            stride=3
        )  # ask daniel

        self.fc = nn.Linear(
            self.output_pool_width * self.output_pool_height * 6, 4)

    def forward(self, x):
        batch_size, frame_size, mffc_nr = x.shape
        x = x.view(batch_size, 1, frame_size, mffc_nr)

        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x
