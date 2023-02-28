import torch
import torch.nn as nn
from .tripletloss import TripletLoss
from .reconstruction_loss import ReconstructionLoss
from .VAE_loss import VAELoss


def get_loss(cfg):
    if cfg.CRITERION.NAME == 'CE':
        loss_function = CrossEntropyLoss()

    elif cfg.CRITERION.NAME == 'MSE':
        loss_function = MSELoss()

    elif cfg.CRITERION.NAME == 'ReconstructionLoss':
        loss_function = ReconstructionLoss()

    elif cfg.CRITERION.NAME == 'VAELoss':
        loss_function = VAELoss()

    else:
        loss_function = TripletLoss(t1=0, t2=0, beta=2)
    return loss_function


class CrossEntropyLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, y_hat, y, *args, **kwargs):
        return {'loss': self.loss_func(y_hat, y, *args, **kwargs)}


class MSELoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss_func = nn.MSELoss(*args, **kwargs)

    def forward(self, y_hat, y, *args, **kwargs):
        return {'loss': self.loss_func(y_hat, y, *args, **kwargs)}
