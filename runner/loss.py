
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


class LabeledDiceLoss(nn.Module):
    def __init__(self):
        super(LabeledDiceLoss, self).__init__()

    def forward(self, predict, gt, labels, is_average=True):
        dice_sum = 0.
        pg = predict * gt
        pp = predict * predict
        gg = gt * gt
        batch_num = predict.shape[0]

        for i in range(batch_num):
            if labels[i] == 1:
                intersection = torch.sum(pg[i])
                union = torch.sum(pp[i]) + torch.sum(gg[i])
                dice = (2. * intersection + 1e-5) / (union + 1e-5)
            else:
                union = torch.sum(predict[i])
                dice = 25. / (union + 25.)
            dice_sum += dice
        if is_average:
            return (batch_num - dice_sum) / batch_num
        else:
            return batch_num - dice_sum


class ClsSegLoss(nn.Module):
    def __init__(self):
        super(ClsSegLoss, self).__init__()

    def forward(self, predict_cls, predict_seg, labels, masks, is_average=True):
        predict_cls = predict_cls.float()
        predict_seg = predict_seg.float()
        labels = labels.float()
        masks = masks.float()

        predict_seg = F.sigmoid(predict_seg)
        prob_seg = []
        gt_mask = []
        label_seg = []
        for index, prob in enumerate(predict_cls):
            if prob >= 0.5:
                prob_seg.append(predict_seg[index])
                gt_mask.append(masks[index])
                label_seg.append(labels[index])

        if is_average:
            cls_loss = F.binary_cross_entropy(predict_cls, labels)
        else:
            cls_loss = F.binary_cross_entropy(predict_cls, labels, reduction='sum')

        if len(prob_seg):
            prob_seg = torch.cat(prob_seg, 0)
            gt_mask = torch.cat(gt_mask, 0)
            loss_fun = LabeledDiceLoss()
            seg_loss = loss_fun(prob_seg, gt_mask, label_seg, is_average=is_average)

            return cls_loss, seg_loss
        else:
            seg_loss = torch.tensor(1e-4).cuda().float()

            return cls_loss, seg_loss
