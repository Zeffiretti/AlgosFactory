import torch
from torch import nn

from .embedding import SinusoidalPositionalEmbedding


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.action_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(self.t_dim),
            nn.Linear(self.t_dim, self.t_dim * 2),
            nn.Mish(),
            nn.Linear(self.t_dim * 2, self.t_dim),
        )
        self.time_mlp = self.time_mlp.to(device)

        input_dim = state_dim + action_dim + self.t_dim
        self.mid_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mid_layers = self.mid_layers.to(device)

        self.output_layer = nn.Linear(hidden_dim, self.action_dim)
        self.output_layer = self.output_layer.to(device)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        time_emb = self.time_mlp(time)
        x = torch.cat([x, state, time_emb], dim=1)
        x = self.mid_layers(x)
        return self.output_layer(x)
