from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def conv_output_dim(input_dim, kernel, stride, padding=0):
    return int(((input_dim + 2 * padding - (kernel - 1) - 1) / stride) + 1)


def pool_output_dim(input_dim, kernel, stride, padding=0):
    return int(((input_dim - kernel + 2 * padding) / stride) + 1)


class SpeechCnn(nn.Module):
    def __init__(self,
                 conv_kernel: Tuple[int, int] = (5, 5),
                 pool_kernel: Tuple[int, int] = (5, 5),
                 pool_stride: int = 3,
                 output_dimension: int = 50
                 ):
        super().__init__()

        # to reduce the dimensions, [50, 100] is fine
        self.output_dimension = output_dimension

        self.conv_kernel_height, self.conv_kernel_width = conv_kernel
        self.pool_kernel_height, self.pool_kernel_width = pool_kernel
        self.pool_stride = pool_stride

        self.output_pool_width = pool_output_dim(
            conv_output_dim(13, self.conv_kernel_width, 1, 0),
            self.pool_kernel_width, self.pool_stride, 0)
        self.output_pool_height = pool_output_dim(
            conv_output_dim(3361, self.conv_kernel_height, 1, 0),
            self.pool_kernel_height, self.pool_stride, 0)

        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=(self.conv_kernel_height, self.conv_kernel_width)
        )

        self.pool = nn.MaxPool2d(
            kernel_size=(self.pool_kernel_height, self.pool_kernel_width),
            stride=self.pool_stride
        )

        self.fc = nn.Linear(
            self.output_pool_width * self.output_pool_height * 6,
            self.output_dimension)

    def forward(self, x):
        batch_size, frame_size, mffc_nr = x.shape
        x = x.view(batch_size, 1, frame_size, mffc_nr)

        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc(x)
        return x
