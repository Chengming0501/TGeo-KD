import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self, args):
        super(CELoss, self).__init__()
        if hasattr(args, 'weight'):
            self.CE_loss = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(args.weight, device=args.device))
        else:
            self.CE_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        loss = self.CE_loss(output, target)

        return loss.mean()