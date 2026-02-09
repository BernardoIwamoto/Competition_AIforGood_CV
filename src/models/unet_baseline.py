import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2)

        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(x1)
        x3 = F.relu(self.enc2(x2))

        x4 = F.interpolate(x3, scale_factor=2, mode="bilinear", align_corners=False)
        x5 = F.relu(self.dec1(x4))

        return self.out(x5)