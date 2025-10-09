import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import CSVLoader, CSVDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

buildin_model = {
    "tiny": [4, "tanh", 6, "leakyrelu", 4, "tanh"],
    "normal": [4, "tanh", 8, "leakyrelu", 8, "relu", 4, "tanh"],
    "huge": [4, "tanh", 8, "tanh", 16, "relu", 20, "leakyrelu", 10, "tanh"]
}

class FCN(nn.Module):
    """FC Network"""
    def __init__(self, in_dim, hidden_params, out_dim=None, device="cpu"):
        """Use params to build a network"""
        super().__init__()
        self.in_dim = in_dim
        self.device = device
        layers = []
        prev = in_dim
        for p in hidden_params:
            if isinstance(p, int):
                # FC layer
                layers.append(nn.Linear(prev, p))
                prev = p
            else:
                p = p.lower()
                # activate function
                # support activate function type: sigmoid, tanh, relu, leakyrelu
                if p == "sigmoid":
                    layers.append(nn.ReLU())
                elif p == "tanh":
                    layers.append(nn.Tanh())
                elif p == "relu":
                    layers.append(nn.ReLU())
                elif p == "leakyrelu":
                    layers.append(nn.LeakyReLU(0.05))
                else:
                    raise ValueError(f"Unknown activate function type: {p}")
        if out_dim is None:
            self.out_dim = prev
        else:
            layers.append(nn.Linear(prev, out_dim))
            layers.append(nn.ReLU())
            self.out_dim = out_dim
        self.model = nn.Sequential(*layers).to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape like batch_size * in_dim
        Returns:
            torch.Tensor: shape like batch_size * out_dim
        """
        assert x.shape[1] == self.in_dim
        x = self.model(x)
        return x

class Embedding(nn.Module):
    def __init__(self, in_dim, embed_dim, device="cpu"):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LeakyReLU(0.05)
        ).to(device=device)
    def forward(self, x):
        assert x.shape[1] == self.in_dim
        x = self.model(x)
        return x
    
class Network(nn.Module):
    def __init__(self, in_dim, embed_dim, hidden_params, num_classes, device="cpu"):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.device = device
        self.embedding = Embedding(in_dim, embed_dim, device=device)
        self.fcn = FCN(embed_dim, hidden_params, num_classes, device=device)
        self.header = nn.Softmax(dim=1).to(device=device)
    def forward(self, x):
        assert x.shape[1] == self.in_dim
        x = self.embedding(x)
        x = self.fcn.forward(x)
        x = self.header(x)
        return x

def parse():
    parser = argparse.ArgumentParser(description="Train and evaluate a simple FCN with CSV data.")
    parser.add_argument("--csv_path", type=str, default="./dataset.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epoch", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument(
        "--criterion",
        type=str,
        default="crossentropy",
        help="supported criterion type: crossentropy, mse, "
    )
    parser.add_argument(
        "--structure",
        type=str,
        default="normal",
        help="Structure of FCN, split by \',\', examples: 32,relu,32,relu"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    # load data
    data_reader = CSVLoader(args.csv_path, sep=",", test_ratio=0.1, device=device)
    X_train, y_train, X_test, y_test = data_reader.load()
    train_dataset = CSVDataset(X_train, y_train)
    test_dataset = CSVDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False)
    # readin the structure of hidden networks
    structure = args.structure
    if structure in buildin_model.keys():
        hidden_params = buildin_model[structure]
    elif len(structure) == 0:
        # empty hidden layer
        hidden_params = []
    else:
        layer_info = structure.split(",")
        hidden_params = []
        for part in layer_info:
            if part.isdigit():
                hidden_params.append(int(part))
            else:
                hidden_params.append(part)
    # build network
    net = Network(
        in_dim=2,
        embed_dim=4,
        hidden_params=hidden_params,
        num_classes=4,
        device=device
    )
    # other hyper-params
    num_epoch = args.num_epoch
    criterion = args.criterion
    if criterion == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    elif criterion == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function type: {criterion}")
    learning_rate = args.learning_rate

    # start training!
    print(f"Network Structure: {net}")
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net = net.to(device)
    net.train()
    for epoch in range(num_epoch):
        total_loss = 0.0
        print("-" * 20 + f" Epoch {epoch} Start " + "-" * 20)
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1} / {num_epoch}"
        )
        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)

            outputs = net.forward(features)
            loss = criterion(outputs, labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        epoch_avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1} / {num_epoch}] Average Loss: {epoch_avg_loss:.5f}")

    # validate
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in test_dataloader:
            features, labels = features.to(device), labels.to(device)
            output = net(features)
            _, predict = torch.max(output, dim=1)
            _, labels = torch.max(labels, dim=1)
            total += labels.shape[0]
            correct += (predict == labels).sum().item()
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")