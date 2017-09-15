import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class BCELoss_seg(nn.Module):
    def __init__(self):
        super(BCELoss_seg, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(inputs_flat, targets_flat)


class weightedBCELoss2d(nn.Module):
    def __init__(self):
        super(weightedBCELoss2d, self).__init__()

    def forward(self, inputs, labels, weights):
        w = weights.view(-1)
        z = inputs.view(-1)
        t = labels.view(-1)
        loss = w * z.clamp(min=0) - w * z * t + w * \
                                                torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / w.sum()
        return loss


class diceAcc(nn.Module):
    def __init__(self, size_average=True):
        self.size_average = size_average
        super(diceAcc, self).__init__()

    def forward(self, inputs, targets):
        size = targets.size(0)
        inputs = inputs.view(size, -1)
        targets = targets.view(size, -1)
        intersection = (inputs * targets)

        res = 2. * (intersection.sum(1) + 1) / (inputs.sum(1) +
                                                targets.sum(1) + 1)
        return res.sum() / size if self.size_average else res


class SoftDiceLoss(nn.Module):
    def __init__(self):  # weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class weightedDiceLoss(nn.Module):
    def __init__(self):
        super(weightedDiceLoss, self).__init__()

    def forward(self, inputs, labels, weights):
        inputs = F.sigmoid(inputs)
        num = labels.size(0)
        w = (weights).view(num, -1)
        w2 = w * w
        m1 = (inputs).view(num, -1)
        m2 = (labels).view(num, -1)
        intersection = (m1 * m2)
        score = 2. * ((w2 * intersection).sum(1) + 1) / \
                ((w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return score


class BCEplusDice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEplusDice, self).__init__()
        self.BCE = BCELoss_seg()
        self.dice = SoftDiceLoss()

    def forward(self, inputs, targets):
        return self.BCE(inputs, targets) + self.dice(inputs, targets)


class weightedBCEplusDice(nn.Module):
    def __init__(self, use_weights=True):
        super(weightedBCEplusDice, self).__init__()
        self.use_weights = use_weights
        self.bce_loss = weightedBCELoss2d()
        self.dice_loss = weightedDiceLoss()

    def forward(self, inputs, targets):
        # compute weights
        batch_size, C, H, W = targets.size()
        if H == 128:
            kernel_size = 11
        elif H == 256:
            kernel_size = 21
        elif H == 512:
            kernel_size = 21
        elif H == 1024:
            kernel_size = 41  # 41
        else:
            raise ValueError('exit at criterion()')

        a = F.avg_pool2d(targets, kernel_size=kernel_size,
                         padding=kernel_size // 2, stride=1)
        ind = a.ge(0.01) * a.le(0.99)
        ind = ind.float()
        weights = Variable(torch.tensor.torch.ones(a.size())).cuda()

        if self.use_weights:
            w0 = weights.sum()
            weights = weights + ind * 2
            w1 = weights.sum()
            weights = weights / w1 * w0

        l = self.bce_loss(inputs, targets, weights) + \
            self.dice_loss(inputs, targets, weights)

        return l