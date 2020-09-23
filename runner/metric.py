
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DiceMetric(nn.Module):
    """Dice Metric Function"""
    def __init__(self, dims=[2, 3, 4]):
        super(DiceMetric, self).__init__()
        self.dims = dims

    def forward(self, predict, gt, activation="sigmoid", is_average=True):
        """
        Args:
            predict(torch tensor):
            gt(torch tensor):
            activation(str): sigmoid or softmax
            is_average(bool):
        Return:
            dice(torch tensor):
        """
        predict = predict.float()
        gt = gt.float()

        if activation == "sigmoid":
            pred = F.sigmoid(predict)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
        elif activation == "softmax":
            pred = torch.argmax(predict, dim=1)

        intersection = torch.sum(pred * gt, dim=self.dims)
        union = torch.sum(pred, dim=self.dims) + torch.sum(gt, dim=self.dims)
        dice = (2. * intersection + 1e-5) / (union + 1e-5)
        dice = dice.mean(0) if is_average else dice.sum(0)

        return dice


def compute_precision_recall_F1(predict, gt, num_class):
    """compute precision, recall and F1"""
    tp, tp_fp, tp_fn = [0.] * num_class, [0.] * num_class, [0.] * num_class
    precision, recall, f1 = [0.] * num_class, [0.] * num_class, [0.] * num_class
    for label in range(num_class):
        t_labels = gt == label
        p_labels = predict == label
        tp[label] += np.sum(t_labels == (p_labels * 2 - 1))
        tp_fp[label] += np.sum(p_labels)
        tp_fn[label] += np.sum(t_labels)
        precision[label] = tp[label] / (tp_fp[label] + 1e-8)
        recall[label] = tp[label] / (tp_fn[label] + 1e-8)
        f1[label] = 2 * precision[label] * recall[label] / (precision[label] + recall[label] + 1e-8)

    return precision, recall, f1

