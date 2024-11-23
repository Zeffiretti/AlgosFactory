import math

import torch


class Embedding(torch.nn.Module):
    def __init__(self, n_nodes, n_dimensions):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(n_nodes, n_dimensions)
        self.positional_embedding = torch.nn.Embedding(n_nodes, n_dimensions)

        self.register_buffer("pos_ids", torch.arange(n_nodes).long())

    def forward(self, node_ids):
        return self.embedding(node_ids)


class SinusoidalPositionalEmbedding(torch.nn.Module):
    def __init__(self, n_dimensions):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.n_dimensions = n_dimensions

    def forward(self, x):
        device = x.device
        half_dim = self.n_dimensions // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
