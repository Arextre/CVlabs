import os
import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from models import Net

def _get_last_conv(model: Net):
    last_resblock = model.net.feature_extractor[-4]
    last_conv = last_resblock.model[-2]
    last_bn = last_resblock.model[-1]
    assert isinstance(last_conv, nn.Conv2d), "Expected Conv2d layer."
    assert isinstance(last_bn, nn.BatchNorm2d), "Expected BatchNorm2d layer."
    return last_conv, last_bn

@torch.no_grad()
def collect_channel_activate_means(
    model: Net,
    dataloader,
    device: torch.device | None = None,
    collect_type: str = "pixel_means",
) -> torch.Tensor:
    
    model.eval()
    last_conv, _ = _get_last_conv(model)

    channel_sum = None
    means_count = {
        "pixel_means": 0,
        "batch_means": 0
    }

    if collect_type == "pixel_means":
        sum_dim = (0, 2, 3)
    elif collect_type == "batch_means":
        sum_dim = (0,)
    else:
        raise ValueError(f"Unknown collect_type: {collect_type}")
    
    def hook_fn(_module, _input, output):
        nonlocal channel_sum, means_count
        out = output.detach()
        b, c, h, w = out.shape
        cur_sum = out.float().sum(dim=sum_dim)  # (C, ) or (C, H, W)
        if channel_sum is None:
            channel_sum = cur_sum.cpu().clone()
        else:
            channel_sum += cur_sum.cpu()
        means_count["pixel_means"] += b * h * w
        means_count["batch_means"] += b
    
    handle = last_conv.register_forward_hook(hook_fn)
    for batch in tqdm(dataloader, desc="Collecting activations", leave=False):
        (img1, img2), _ = batch
        # resolve device lazily to avoid circular import dependency on main
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img1, img2 = img1.to(device), img2.to(device)
        img1 = img1.unsqueeze(1)  # add channel dim
        img2 = img2.unsqueeze(1)
        _ = model(img1, img2)
    
    handle.remove()
    if channel_sum is not None and means_count[collect_type] > 0:
        channel_means = channel_sum / float(means_count[collect_type])
        return channel_means  # (C, )
    else:
        return torch.empty(0)

@torch.no_grad()
def _prune_model(
    model: Net,
    dataloader: torch.utils.data.DataLoader,
    num_prune: int,
    device: torch.device | None = None
):
    # collect activations (device resolved inside)
    channel_means = collect_channel_activate_means(model, dataloader, device=device)
    if channel_means.numel() == 0:
        print("No activations collected; skipping pruning.")
        return []
    chnl_to_prune = channel_means.numel()
    assert 0 < num_prune < chnl_to_prune, f"num_prune should be in (0, {chnl_to_prune})"

    last_conv, last_bn = _get_last_conv(model)
    prune_idx = torch.argsort(channel_means)[:num_prune].tolist()

    results = []
    with torch.no_grad():

        for idx in prune_idx:
            last_conv.weight.data[idx] = 0
            if last_conv.bias is not None:
                last_conv.bias.data[idx] = 0
            if hasattr(last_bn, "weight") and last_bn.weight is not None:
                last_bn.weight.data[idx] = 0
            if hasattr(last_bn, "bias") and last_bn.bias is not None:
                last_bn.bias.data[idx] = 0

            # delay import of validate to runtime to avoid circular import
            from main import validate
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            acc = validate(model, dataloader, device)
            results.append(acc)
            print(f"Pruned channels: {prune_idx}")
            print(f"Accuracy after pruning: {acc*100:.2f}%")
    
    print(f"Pruning results: {results}")
    return results

def _auto_grid(n: int) -> tuple[int, int]:
    rows = int(math.floor(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    if rows * cols < n:
        cols += 1
    return rows, cols

def plot_feature_map_grid(
    feature_map: torch.Tensor,
    path: str="./notebook/feature_maps.png"
):
    assert feature_map.ndim == 3, "Expected feature_map to have 3 dimensions (C,H,W)."
    channel, h, w = feature_map.shape
    rows, cols = _auto_grid(channel)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(rows, cols)
    vmin = feature_map.min().item()
    vmax = feature_map.max().item()
    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.set_axis_off()
        if idx < channel:
            img = feature_map[idx].numpy()
            ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"Channel_{idx}", fontsize=8)
    fig.suptitle("Channel-wise Mean Feature Maps")
    fig.tight_layout()
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
        

def prune_model(
        model: Net,
        dataloader: torch.utils.data.DataLoader,
        num_prune: int,
        feature_map_path: str | None = None,
        device: torch.device | None = None,
):
    feature_map = collect_channel_activate_means(
        model,
        dataloader,
        device,
        collect_type="batch_means",
    )
    if feature_map_path is not None:
        plot_feature_map_grid(feature_map)

    results = _prune_model(model, dataloader, num_prune, device)
    return results