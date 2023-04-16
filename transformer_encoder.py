import torch
import torch.nn as nn
import numpy as np

from attention import MultiHeadAttention
from utils import AddNorm


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = MultiHeadAttention()
        self.add_norm = AddNorm()
        self.layers = nn.ModuleList([])






