from torch import nn
import torch.nn.functional as F


class BCELoss_logits(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss_logits, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        probs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        num = targets.size(0)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class BCE_plus_Dice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCE_plus_Dice, self).__init__()
        self.BCE = BCELoss_logits(weight, size_average)
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        return self.BCE(inputs, targets) + self.dice(inputs, targets)
