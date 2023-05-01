import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (pred>0).detach()
        
        diff = target[valid_mask] - pred[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (pred>0).detach()
        
        diff = target[valid_mask] - pred[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class MaskedScaleInvariantLoss(nn.Module):
    def __init__(self):
        super(MaskedScaleInvariantLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (pred>0).detach()
        
        diff = target[valid_mask] - pred[valid_mask]
        
        scale_invariant_term = (diff.sum() ** 2) / float(diff.size()[0]) ** 2 
        
        self.loss = (diff ** 2).mean() - scale_invariant_term
        return self.loss