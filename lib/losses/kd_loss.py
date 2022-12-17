import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, args):
        super(KDLoss, self).__init__()
        self.temperature = args.temperature
        self.loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, output, target):
        # breakpoint()
        KD_loss = self.loss(F.log_softmax(output/self.temperature, dim=1),
                             # F.softmax(target/self.temperature, dim=1))
                            target / self.temperature)
        return KD_loss