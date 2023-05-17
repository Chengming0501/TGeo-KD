import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, args):
        super(KDLoss, self).__init__()
        self.temperature = args.temperature
        self.loss = nn.KLDivLoss(reduction='none')
        self.CE_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target, gt, alpha):
        KD_loss = self.loss(F.log_softmax(output/self.temperature, dim=1),
                            F.softmax(target / self.temperature, dim=1))
        CE_loss = self.CE_loss(output, gt)
        loss = (1 - alpha) * CE_loss + alpha * KD_loss.sum(1)
        return loss.mean()
