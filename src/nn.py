import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=5)

        self.fc1 = nn.Linear(12 * 12 * 7, 256)
        self.fc2 = nn.Linear(256, 16)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)

        x = x.view(-1)

        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def load(self):
        try:
            self.load_state_dict(torch.load('state_dict.pth'))
        except FileNotFoundError:
            print('State dict not found')

    def store(self):
        try:
            torch.save(self.state_dict(), f='state_dict.pth')
        except OSError:
            print('Wasn\'t able to save State dict')


# TESTING
if __name__ == '__main__':  # Only execute if called
    import torch

    torch.tensor()
