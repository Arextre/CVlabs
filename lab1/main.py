import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import CSVLoader, CSVDataset

if __name__ == "__main__":
    # default: dataset.csv next to this script
    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, 'dataset.csv')

    print(f"Loading and splitting: {csv_path}")
    model = train_and_evaluate(csv_path)