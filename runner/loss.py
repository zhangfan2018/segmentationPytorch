
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss Function"""
    def __init__(self, dims=[2, 3, 4]):
        super(DiceLoss, self).__init__()
        self.dims = dims

    def forward(self, predict, gt, activation="sigmoid", is_average=True):
        """
        Args:
            predict(torch tensor):
            gt(torch tensor):
            activation(str): sigmoid or softmax
            is_average(bool):
        Return:
            dice_loss(torch tensor):
        """
        predict = predict.float()
        gt = gt.float()
        if activation == "softmax":
            probability = F.softmax(predict, dim=1)
        elif activation == "sigmoid":
            probability = F.sigmoid(predict)

        intersection = torch.sum(probability*gt, dim=self.dims)
        union = torch.sum(probability*probability, dim=self.dims) + torch.sum(gt*gt, dim=self.dims)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice_loss = 1 - dice.mean(1)
        dice_loss = dice_loss.mean() if is_average else dice_loss.sum()

        return dice_loss


class BCELoss(nn.Module):
    """BCE loss Function"""
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, predict, gt, activation="sigmoid", is_average=True):
        """
        Args:
            predict(torch tensor):
            gt(torch tensor):
            activation(str): sigmoid or softmax
            is_average(bool):
        Return:
            bce_loss(torch tensor):
        """
        predict = predict.float()
        gt = gt.float()
        if activation == "softmax":
            probability = F.softmax(predict, dim=1)
        elif activation == "sigmoid":
            probability = F.sigmoid(predict)
        bce_loss = F.binary_cross_entropy(probability, gt, size_average=is_average)

        return bce_loss
