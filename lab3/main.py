import os
import gc
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Net
from prune_module import prune_model
from utils import ComposeMnistDataset, train, validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported else torch.float32

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--select_scale",
        type=float,
        default=0.1,
        help="Scale of training data selected from MNIST"
    )
    parser.add_argument(
        "--embed_channels",
        type=int,
        default=9,
        help="Number of embedding channels"
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=32,
        help="Number of hidden channels"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="./notebook/logs",
        help="Path to save logs"
    )
    parser.add_argument(
        "--num_prune",
        type=int,
        default=4,
        help="Number of channels to prune"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")
    print(f"Log path: {args.log_path}")
    print(f"Arguments: {args}")

    # writer = SummaryWriter(log_dir=args.log_path)

    train_dataset = ComposeMnistDataset(scale=args.select_scale,
                                        is_train=True,
                                        dtype=DTYPE,
                                        device=DEVICE)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    test_dataset = ComposeMnistDataset(scale=args.select_scale,
                                       is_train=False,
                                       dtype=DTYPE,
                                       device=DEVICE)
    test_dataset = test_dataset.to(device=DEVICE)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    model = Net(embed_channels=args.embed_channels,
                hidden_channels=args.hidden_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device=DEVICE, dtype=DTYPE)
    criterion = criterion.to(device=DEVICE, dtype=DTYPE)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Model structure:\n{model}")

    for epoch in range(args.num_epoch):
        print(f"Epoch {epoch + 1} / {args.num_epoch}")
        train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            writer=None,
            device=DEVICE
        )

    fin_acc = validate(model, test_dataloader, DEVICE)
    
    results = [fin_acc]  # Store accuracy before pruning
    prune_results = prune_model(
        model,
        test_dataloader,
        num_prune=args.num_prune,
        feature_map_path=os.path.join(args.log_path, "feature_map.png"),
        device=DEVICE
    )
    results.extend(prune_results)
    
    writer = SummaryWriter(log_dir=os.path.join(args.log_path, "prune_logs"))
    for i, acc in enumerate(results):
        writer.add_scalar("Prune/Accuracy", acc, global_step=i)
    writer.close()