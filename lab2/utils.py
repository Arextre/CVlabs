import gc
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class ComposeMnistDataset(Dataset):
    def __init__(
            self,
            scale: float=0.1,
            is_train: bool=True,
            dtype: torch.dtype=torch.float32,
            path: str='./notebook/data',
            seed: int=13
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
        mnist_data = mnist_dataset.data.to(dtype=dtype)
        mnist_labels = mnist_dataset.targets.to(dtype=dtype)
        
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
                    label = torch.tensor([0, 1], dtype=dtype)

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
            label = torch.tensor([1, 0], dtype=dtype)

            self.data.append((img1, img2))
            self.labels.append(label)
    
        print(f"Total number of samples: {len(self.labels)}")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx], self.labels[idx]
                
            
        
        