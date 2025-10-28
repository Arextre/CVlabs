import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class ComposeMnistDataset(Dataset):
    def __init__(self, scale: float=0.1, is_train: bool=True, path: str='./notebook/data', seed=13):
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
        mnist_data = mnist_dataset.data
        mnist_labels = mnist_dataset.targets
        
        buc = [[] for _ in range(10)]   # 10 buckets for 10 classes
        for i in range(mnist_data.shape[0]):
            label = int(mnist_labels[i].item())
            buc[label].append(mnist_data[i])
        selected_data = []
        selected_labels = []
        np.random.seed(seed)
        for i in range(10):
            n_select = int(round(len(buc[i]) * scale))
            np.random.shuffle(buc[i])
            selected_data.extend(buc[i][: n_select])
            selected_labels.extend([i] * n_select)
        print(f"selected_data length: {len(selected_data)}")
        self.data = []
        self.labels = []
        for i in range(len(selected_data)):
            for j in range(len(selected_data)):
                if i == j:
                    continue
                img1 = selected_data[i] # (28, 28)
                img2 = selected_data[j]
                img = torch.stack([img1, img2], dim=0) # (2, 28, 28)
                label = 1 if selected_labels[i] == selected_labels[j] else 0
                self.data.append(img)
                self.labels.append(label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx], self.labels[idx]
                
            
        
        