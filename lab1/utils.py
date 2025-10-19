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
                 seed: int = 1113):
        self.csv_path = csv_path
        self.sep = sep
        self.test_ratio = test_ratio
        self.seed = seed
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
        data = torch.from_numpy(df.values).to(dtype=torch.float32)
        buc = [[] for _ in range(4)] # 4 classes
        for i in range(data.shape[0]):
            buc[int(data[i, -1]) - 1].append(data[i]) # split by label
        
        np.random.seed(self.seed)
        X_train, y_train, X_test, y_test = [], [], [], []
        for bucket in buc:
            # shuffle samples in each bucket
            np.random.shuffle(bucket)
            bucket = torch.stack(bucket, dim=0)
            train_samples = int(bucket.shape[0] * (1 - self.test_ratio))
            # split train/validate
            train, test = bucket[:train_samples], bucket[train_samples:]
            X_train.append(train[:, :-1])
            y_train.append(train[:, -1])
            X_test.append(test[:, :-1])
            y_test.append(test[:, -1])
        # to tensor
        X_train = torch.cat(X_train, dim=0)
        y_train = torch.cat(y_train, dim=0)
        X_test = torch.cat(X_test, dim=0)
        y_test = torch.cat(y_test, dim=0)
        
        # trans labels to one-hot
        def to_onehot(label: torch.Tensor) -> torch.Tensor:
            label = label.to(dtype=torch.int64) - 1
            label = F.one_hot(label)
            label = label.to(dtype=torch.float32)
            return label
        y_train = to_onehot(y_train[:])
        y_test = to_onehot(y_test[:])
        return X_train, y_train, X_test, y_test
    
class CSVDataset(Dataset):
    """Loading data from CSV files using CSVLoader and can be used with torch DataLoader.
    """
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
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
        