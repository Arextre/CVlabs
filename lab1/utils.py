import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset

class CSVLoader:
    """Simple CSV loader that handles header/missing values via pandas when available,
    otherwise falls back to numpy.loadtxt.

    Usage:
        loader = CSVLoader("dataset.csv")
        X_train, y_train, X_test, y_test = loader.load()
    """

    def __init__(self,
                 csv_path: str,
                 sep: str = ",",
                 test_ratio: float = 0.1,
                 seed: int = 42,
                 device="cpu"):
        self.csv_path = csv_path
        self.sep = sep
        self.test_ratio = test_ratio
        self.seed = seed
        self.device = device
        self.trainData = None
        self.trainLabel = None
        self.testData = None
        self.testLabel = None
    
    def load(self):
        """
        Returns:
            X_train, y_train, X_test, y_test (torch.Tensor)
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path, sep=self.sep, header=None)
        data = torch.from_numpy(df.values).to(device=self.device,
                                              dtype=torch.float32)
        # shuffle the data
        indices = torch.randperm(data.shape[0])
        data = data[indices]
        # split data to train and validate part
        def to_onehot(label: torch.Tensor):
            label = label.to(dtype=torch.int64) - 1
            label = F.one_hot(label)
            label = label.to(dtype=torch.float32)
            return label
        total_samples = data.shape[0]
        train_samples = int(total_samples * (1 - self.test_ratio))
        train_part = data[:train_samples]
        test_part = data[train_samples:]
        X_train = train_part[:, :-1]
        y_train = to_onehot(train_part[:, -1])
        X_test = test_part[:, :-1]
        y_test = to_onehot(test_part[:, -1])
        return X_train, y_train, X_test, y_test
    
class CSVDataset(Dataset):
    """Loading data from CSV files using CSVLoader and can be used with torch DataLoader.
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.len = data.shape[0]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        assert index < self.len
        X = self.data[index]
        y = self.labels[index]
        return X, y
        