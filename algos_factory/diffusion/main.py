import logging

import torch

from algos_factory.diffusion import Diffusion

log = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

log.info("Using device: %s", device)

x = torch.randn(256, 2).to(device)
state = torch.randn(256, 11).to(device)
model = Diffusion(loss_type="l2", state_dim=11, action_dim=2, hidden_dim=256, diffusion_steps=100, device=device)
action = model(state)

loss = model.loss(x, state)

log.info("Loss: %s", loss.item())
