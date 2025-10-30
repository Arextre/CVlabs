import gc
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import Net
from utils import ComposeMnistDataset

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
    return parser.parse_args()

def validate(model, dataloader, device=DEVICE) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):

            (img1, img2), label = batch
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            img1 = img1.unsqueeze(1)  # (B, 1, 28, 28)
            img2 = img2.unsqueeze(1)

            out = model(img1, img2)
            predicted = torch.argmax(out, 1)
            labels = torch.argmax(label, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            del img1, img2, label, out, predicted, labels

        gc.collect()
        torch.cuda.empty_cache()
    return correct / total if total > 0 else 0

def train(model, optimizer, criterion, train_dataloader, test_dataloader, writer=None, device=DEVICE):
    model.train()
    epoch_loss = 0.0
    global_step = 0
    progress_bar = tqdm(train_dataloader, desc="Training", leave=False)
    for idx, batch in enumerate(progress_bar, start=1):

        optimizer.zero_grad()

        (img1, img2), label = batch
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        img1 = img1.unsqueeze(1)  # (B, 1, 28, 28)
        img2 = img2.unsqueeze(1)

        out = model(img1, img2)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

        if idx % 10 == 0 and writer is not None:

            acc = validate(model, test_dataloader, device)
            model.train()
            writer.add_scalar("Train/Loss", loss.item(), global_step=global_step)
            writer.add_scalar("Train/Accuracy", acc, global_step=global_step)
            global_step += 1

            del img1, img2, label, out, loss
            gc.collect()
            torch.cuda.empty_cache()

    acc = validate(model, test_dataloader, device)
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"----- Training finished, Epoch Loss: {avg_loss:.4f}, Accuracy: {acc * 100:.2f}% -----")
    
    if writer is not None:
        writer.add_scalar("Train/Accuracy", acc, global_step=global_step)

def prune_model(model: Net, num_prune, activations):
    last_resblock = model.net.feature_extractor[-4]
    last_conv = last_resblock.model[-2]


if __name__ == "__main__":
    args = parse()
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")
    print(f"Log path: {args.log_path}")
    print(f"Arguments: {args}")

    writer = SummaryWriter(log_dir=args.log_path)

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
        train(model, optimizer, criterion, train_dataloader, test_dataloader, writer)
        # accuracy = validate(model, test_dataloader)
        # print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    
    # --- Example: prune by activations using forward_hook (uncomment to run) ---
    # from_this = prune_model_with_hook(model, test_dataloader, num_prune=5, device=DEVICE)
    # acc_after = validate(model, test_dataloader, device=DEVICE)
    # print(f"Accuracy after pruning 5 channels: {acc_after * 100:.2f}%")

    # --- Example: collect and plot mean feature maps of last_conv (before pruning) ---
    # mean_maps = collect_last_conv_mean_feature_maps(model, test_dataloader, device=DEVICE)
    # plot_feature_maps_grid(
    #     mean_maps,
    #     save_path=os.path.join(args.log_path, 'mean_feature_maps.png'),
    #     title='Last Conv Mean Feature Maps (test set)'
    # )

    # --- Example: sweep K and plot Accuracy vs K ---
    # results = prune_k_sweep_and_eval(model, test_dataloader, device=DEVICE)
    # plot_prune_curve(
    #     results,
    #     save_path=os.path.join(args.log_path, 'prune_curve.png'),
    #     title='Accuracy vs Pruned Channels K'
    # )
    writer.close()
