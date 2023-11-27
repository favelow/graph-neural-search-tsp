import dgl.nn
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, G=None):
        if G is not None:
            y = self.module(G, x).view(G.number_of_nodes(), -1)
        else:
            y = self.module(x)
        return x + y


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, hidden_dim):
        super().__init__()

        self.message_passing = SkipConnection(
            dgl.nn.GATConv(embed_dim, embed_dim // n_heads, n_heads)
        )

        self.feed_forward = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
         