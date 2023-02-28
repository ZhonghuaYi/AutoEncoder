import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, kld_weight):
        recons, input, mu, log_var = pred
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss

        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}
