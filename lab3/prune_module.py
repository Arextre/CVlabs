import os
import math
import copy
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from models import Net
from main import validate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Pruning utilities (last_conv via forward_hook) ----
def _get_last_conv_and_bn(model: Net):
    """Return the last conv in the last ResidualBlock and its following BN.

    Net structure reminder:
      model.net.feature_extractor = nn.Sequential(
        [0] Conv2d,
        [1] MaxPool2d,
        [2] ResidualBlock,
        [3] MaxPool2d,
        [4] ResidualBlock,   # last ResidualBlock
        [5] Flatten,
        [6] Linear,
        [7] LeakyReLU,
      )
    ResidualBlock.model = nn.Sequential(
        [0] Conv2d,
        [1] BatchNorm2d,
        [2] Dropout,
        [3] ReLU,
        [4] Conv2d,          # last conv inside block
        [5] BatchNorm2d,     # BN after last conv
    )
    """
    last_resblock = model.net.feature_extractor[-4]
    last_conv = last_resblock.model[-2]
    last_bn = last_resblock.model[-1]
    assert isinstance(last_conv, nn.Conv2d), "Expected Conv2d at last_resblock.model[-2]"
    assert isinstance(last_bn, nn.BatchNorm2d), "Expected BatchNorm2d at last_resblock.model[-1]"
    return last_conv, last_bn

@torch.no_grad()
def collect_last_conv_channel_means(model: Net, dataloader: DataLoader, device: torch.device = DEVICE) -> torch.Tensor:
    """Collect per-channel mean activations of the last_conv over the entire dataloader.

    Returns a 1D tensor of shape (C,) with mean activation per output channel.
    Uses a forward_hook and accumulates sums to avoid storing all activations.
    """
    model.eval()
    last_conv, _ = _get_last_conv_and_bn(model)

    channel_sum = None  # torch.Tensor (C,)
    pixel_count = 0     # scalar int for total elements per channel (N*H*W accumulated)

    def hook_fn(_module, _input, output):
        nonlocal channel_sum, pixel_count
        # output: (B, C, H, W)
        out = output.detach()
        B, C, H, W = out.shape
        # sum over batch and spatial dims -> (C,)
        cur_sum = out.float().sum(dim=(0, 2, 3))
        if channel_sum is None:
            channel_sum = cur_sum.cpu()
        else:
            channel_sum += cur_sum.cpu()
        pixel_count += int(B * H * W)

    handle = last_conv.register_forward_hook(hook_fn)

    for (img1, img2), _label in dataloader:
        # Inputs in this project are (B, 28, 28); model expects (B, 1, 28, 28)
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        # Data was already created on the target device; avoid redundant .to()
        _ = model(img1, img2)

    handle.remove()

    if channel_sum is None or pixel_count == 0:
        # No data; return empty tensor
        return torch.empty(0)

    channel_mean = channel_sum / float(pixel_count)
    return channel_mean  # (C,)

def prune_model_with_hook(model: Net, dataloader: DataLoader, num_prune: int, device: torch.device = DEVICE) -> torch.Tensor:
    """Prune the lowest-activation output channels of last_conv using forward_hook stats.

    Steps:
      1) Collect per-channel mean activations of last_conv over dataloader.
      2) Sort ascending; take the first `num_prune` channels.
      3) Zero corresponding Conv2d weights/bias and BatchNorm2d gamma/beta.

    Returns:
      torch.Tensor: The per-channel mean activation vector (before pruning), shape (C,).
    """
    # 1) Collect activations
    means = collect_last_conv_channel_means(model, dataloader, device)
    if means.numel() == 0:
        print("[prune] No activations collected; skip pruning.")
        return means

    # 2) Determine prune indices
    C = means.numel()
    assert 0 < num_prune < C, f"num_prune must be in [1, {C-1}]"
    prune_indices = torch.argsort(means)[:num_prune]

    # 3) Zero Conv and BN parameters for selected channels
    last_conv, last_bn = _get_last_conv_and_bn(model)

    with torch.no_grad():
        # Zero conv out-channels
        last_conv.weight.data[prune_indices] = 0
        if last_conv.bias is not None:
            last_conv.bias.data[prune_indices] = 0

        # Also zero BN gamma (weight) and beta (bias) so outputs stay zero in eval
        if hasattr(last_bn, 'weight') and last_bn.weight is not None:
            last_bn.weight.data[prune_indices] = 0
        if hasattr(last_bn, 'bias') and last_bn.bias is not None:
            last_bn.bias.data[prune_indices] = 0

    print(f"[prune] Pruned channels (count={num_prune}): {prune_indices.tolist()}")
    return means

@torch.no_grad()
def collect_last_conv_mean_feature_maps(model: Net, dataloader: DataLoader, device: torch.device = DEVICE) -> torch.Tensor:
    """Collect average feature maps (C,H,W) of last_conv over the entire dataloader.

    We average over the dataset (batch dimension) only, keeping spatial dimensions.
    Returns: torch.Tensor of shape (C,H,W) on CPU.
    """
    model.eval()
    last_conv, _ = _get_last_conv_and_bn(model)

    sum_map = None  # (C,H,W) on CPU
    count_imgs = 0

    def hook_fn(_m, _inp, out):
        nonlocal sum_map, count_imgs
        # out: (B,C,H,W)
        out = out.detach().float()  # move to float for stable accumulation
        B, C, H, W = out.shape
        batch_sum = out.sum(dim=0).cpu()  # (C,H,W)
        if sum_map is None:
            sum_map = batch_sum.clone()
        else:
            sum_map += batch_sum
        count_imgs += B

    handle = last_conv.register_forward_hook(hook_fn)

    for (img1, img2), _label in dataloader:
        # (B,28,28) -> (B,1,28,28)
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
        _ = model(img1, img2)

    handle.remove()

    if sum_map is None or count_imgs == 0:
        return torch.empty(0)

    mean_map = sum_map / float(count_imgs)
    return mean_map  # (C,H,W)

def _auto_grid(P: int) -> tuple[int, int]:
    """Pick a near-square grid (rows, cols) for P images."""
    rows = int(math.floor(math.sqrt(P)))
    cols = int(math.ceil(P / rows))
    # ensure rows*cols >= P
    while rows * cols < P:
        rows += 1
    return rows, cols

def plot_feature_maps_grid(mean_maps: torch.Tensor, save_path: str | None = None, title: str = "Mean feature maps"):
    """Plot a grid of (C,H,W) mean maps.

    Args:
      mean_maps: torch.Tensor(C,H,W) on CPU
      save_path: optional file path to save figure
    """
    assert mean_maps.ndim == 3, "mean_maps must be (C,H,W)"
    C, H, W = mean_maps.shape
    rows, cols = _auto_grid(C)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(rows, cols)
    vmin = float(mean_maps.min())
    vmax = float(mean_maps.max())
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis('off')
        if idx < C:
            img = mean_maps[idx].numpy()
            ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f"c{idx}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)

def prune_k_sweep_and_eval(model: Net, dataloader: DataLoader, device: torch.device = DEVICE) -> list[tuple[int, float]]:
    """Compute accuracy after pruning K=1..C-1 least-active channels (fresh copy each K).

    Returns list of (K, accuracy). Activations are measured once on the unpruned model.
    """
    # Gather per-channel scalar means to rank channels
    means = collect_last_conv_channel_means(model, dataloader, device)
    if means.numel() == 0:
        return []
    C = means.numel()
    order = torch.argsort(means)  # ascending: least active first

    results: list[tuple[int, float]] = []
    for K in range(1, C):
        # fresh copy so each point is exactly K pruned from baseline
        m_copy = copy.deepcopy(model)
        last_conv, last_bn = _get_last_conv_and_bn(m_copy)
        idx = order[:K]
        with torch.no_grad():
            last_conv.weight.data[idx] = 0
            if last_conv.bias is not None:
                last_conv.bias.data[idx] = 0
            if hasattr(last_bn, 'weight') and last_bn.weight is not None:
                last_bn.weight.data[idx] = 0
            if hasattr(last_bn, 'bias') and last_bn.bias is not None:
                last_bn.bias.data[idx] = 0
        acc = validate(m_copy, dataloader, device)
        results.append((K, acc))
    return results

def plot_prune_curve(results: list[tuple[int, float]], save_path: str | None = None, title: str = "Accuracy vs K (pruned)"):
    if not results:
        return
    Ks, Accs = zip(*results)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(Ks, Accs, marker='o')
    ax.set_xlabel('K (number of pruned channels)')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)