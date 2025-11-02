import gc
import random
import numpy as np
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class ComposeMnistDataset(Dataset):
    def __init__(
        self,
        scale: float=0.1,
        is_train: bool=True,
        dtype: torch.dtype=torch.float32,
        device: torch.device=torch.device('cpu'),
        path: str='./notebook/data',
        seed: int=13,
    ):
        """Compose MNIST dataset from training sets
        Args:
            scale (float): select scale of training data from MNIST
            path (str): Path to download MNIST dataset
            seed (int): Random seed for selection
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        mnist_dataset = datasets.MNIST(root=path,
                                       train=is_train,
                                       download=True,
                                       transform=transform)
        mnist_data = mnist_dataset.data.to(device=device, dtype=dtype)
        mnist_labels = mnist_dataset.targets.to(device=device, dtype=dtype)

        buc = [[] for _ in range(10)]   # 10 buckets for 10 classes
        for i in range(mnist_data.shape[0]):
            label = int(mnist_labels[i].item())
            buc[label].append(mnist_data[i])
        selected_data = [[] for _ in range(10)]
        np.random.seed(seed)
        for i in range(10):
            n_select = int(round(len(buc[i]) * scale))
            np.random.shuffle(buc[i])
            selected_data[i].extend(buc[i][: n_select])

        self.data = []
        self.labels = []

        label_counter_1 = 0
        # for label == 1
        for i in range(10):
            buc_size = len(selected_data[i])
            for j in range(buc_size):
                for k in range(j + 1, buc_size):
                    img1 = selected_data[i][j]
                    img2 = selected_data[i][k]
                    label = torch.tensor([0, 1], device=device, dtype=dtype)

                    self.data.append((img1, img2))
                    self.labels.append(label)
                    label_counter_1 += 1
        
        print(f"Number of positive samples (label==1): {len(self.labels)}")
        # for label == 0, use Downsampling to limit the number of negative samples
        random.seed(seed)
        for _ in range(label_counter_1):
            label_i, label_j = random.sample(range(10), 2)
            assert label_i != label_j, "label_i and label_j must be different"
            buc_size_i = len(selected_data[label_i])
            buc_size_j = len(selected_data[label_j])
            idx_i = random.randint(0, buc_size_i - 1)
            idx_j = random.randint(0, buc_size_j - 1)

            img1 = selected_data[label_i][idx_i]
            img2 = selected_data[label_j][idx_j]
            label = torch.tensor([1, 0], device=device, dtype=dtype)

            self.data.append((img1, img2))
            self.labels.append(label)
    
        print(f"Total number of samples: {len(self.labels)}")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx], self.labels[idx]
                
    def to(self, device: torch.device):
        """Move dataset to specified device

        Warning:
            This operation is slow and may consume a lot of memory.
        """
        for i in range(len(self.data)):
            img1, img2 = self.data[i]
            self.data[i] = (img1.to(device), img2.to(device))
        for i in range(len(self.labels)):
            self.labels[i] = self.labels[i].to(device)
        return self




def validate(
    model,
    dataloader,
    device: torch.device | None = None
) -> float:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def train(
    model,
    optimizer,
    criterion,
    train_dataloader,
    test_dataloader,
    writer=None,
    device: torch.device | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(f"----- Training finished, Epoch Loss: {avg_loss:.4f}, "
          f"Accuracy: {acc * 100:.2f}% -----")
    
    if writer is not None:
        writer.add_scalar("Train/Accuracy", acc, global_step=global_step)