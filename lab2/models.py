import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union

class Embedding(nn.Module):
    def __init__(self, in_channels: int=1, out_channels: int=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-3] == self.in_channels,
            f"Expected input with {self.in_channels} channels, "
            f"but got {x.shape[-3]} channels."
        )
        return self.model(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.head = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-3] == self.channels,
            f"Expected input with {self.channels} channels, "
            f"but got {x.shape[-3]} channels."
        )
        residual = x
        out = self.model(x)
        out += residual
        out = self.head(out)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self, in_channels: int=3, hidden_channels: int=32, feature_dim: int=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),  # (B, hidden_channels, 28, 28)
            nn.MaxPool2d(2, 2),  # (B, hidden_channels, 14, 14)
            ResidualBlock(hidden_channels),
            nn.MaxPool2d(2, 2),  # (B, hidden_channels, 7, 7)
            ResidualBlock(hidden_channels),   # (B, hidden_channels, 7, 7)
            nn.Flatten(),  # (B, hidden_channels * 7 * 7)
            nn.Linear(hidden_channels * 7 * 7, feature_dim),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        assert x1.shape == x2.shape, "Input tensors must have the same shape"
        out1 = self.feature_extractor(x1)
        out2 = self.feature_extractor(x2)
        diff = torch.abs(out1 - out2)
        combined = torch.cat([out1, diff], dim=-1)
        return self.head(combined)
    
class Net(nn.Module):
    def __init__(self, embed_channels: int=18, hidden_channels: int=32):
        super().__init__()
        self.embedding = Embedding(out_channels=embed_channels)
        self.net = SiameseNetwork(in_channels=embed_channels,
                                  hidden_channels=hidden_channels)
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        emb_x1 = self.embedding(x1)  # (B, embed_channels, 28, 28)
        emb_x2 = self.embedding(x2)  # (B, embed_channels, 28, 28)
        out = self.net(emb_x1, emb_x2)  # (B, 2)
        return out
