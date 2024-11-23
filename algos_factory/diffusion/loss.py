import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class WeightedLoss(nn.Module):
    def forward(self, pred, targ, weighted=1.0):
        """pred, targ : [batch_size, action_dim]"""
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weighted).mean()
        return weighted_loss

    def _loss(self, pred, targ):
        raise NotImplementedError


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}
