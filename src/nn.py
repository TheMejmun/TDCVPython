import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)  # 64x64 => 60x60 => 30x30
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=3)  # 30x30 => 28x28 => 14x14
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3)  # 14x14 => 12x12 => 6x6

        self.fc1 = nn.Linear(6 * 6 * 16, 256)
        self.fc2 = nn.Linear(256, 16)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv3(x)), kernel_size=2)

        x = x.view(-1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x


# TESTING
if __name__ == '__main__':  # Only execute if called
    import torch

    torch.tensor()
