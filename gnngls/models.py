import dgl.nn
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

