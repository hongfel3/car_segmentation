from torch import nn
import torch.nn.functional as F


class BCELoss_logits(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss_logits, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets):
        probs = F.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)
