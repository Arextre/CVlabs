import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.modules):
    def __init__(self):
        super().__init__()
        