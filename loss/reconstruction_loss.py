import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_generate, x_label):
        loss = torch.mean((x_generate - x_label) ** 2)
        return {'loss': loss}
