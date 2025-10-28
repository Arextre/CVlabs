import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import ComposeMnistDataset

if __name__ == "__main__":
    train_dataset = ComposeMnistDataset(scale=0.1, is_train=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1024,
                                  shuffle=True,
                                  num_workers=2)
    
    test_dataset = ComposeMnistDataset(scale=0.1, is_train=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1024,
                                 shuffle=False,
                                 num_workers=2)
    