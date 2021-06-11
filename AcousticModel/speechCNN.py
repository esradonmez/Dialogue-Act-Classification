import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 6, 5) # i'm not sure about these
        self.pool = nn.MaxPool2d(13, 13) # ask daniel
        self.fc = nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.fc(x)
        return x