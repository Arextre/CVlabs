import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import CSVLoader, CSVDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse():
    parser = argparse.ArgumentParser(description="Train and evaluate a simple FCN with CSV data.")
    parser.add_argument("--csv_path", type=str, default="./dataset.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--structure", type=str, default="-1")
    

if __name__ == "__main__":
    pass