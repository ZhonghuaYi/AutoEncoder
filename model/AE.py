import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAutoEncoder(nn.Module):
    def __init__(self, in_channels=28*28, latent_dim=10, encoder_hidden_dim=None, decoder_hidden_dim=None):
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.out_channels = in_channels

        if self.encoder_hidden_dim is None:
            self.encoder_hidden_dim = [512, 256, 128, 64]
        if self.decoder_hidden_dim is None:
            self.decoder_hidden_dim = [64, 128, 256, 512]

        self.encoder = []
        for h_dim in self.encoder_hidden_dim:
            self.encoder.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU(True),
                )
            )
            in_channels = h_dim

        self.encoder.append(
            nn.Sequential(
                nn.Linear(in_channels, latent_dim),
                nn.LeakyReLU(True),
            )
        )
        self.encoder = nn.Sequential(*self.encoder)

        self.decoder = []
        in_channels = latent_dim
        for h_dim in self.decoder_hidden_dim:
            self.decoder.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU(True),
                )
            )
            in_channels = h_dim

        self.decoder.append(
            nn.Sequential(
                nn.Linear(in_channels, self.out_channels),
                nn.Sigmoid(),
            )
        )
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, -1)
        h = self.encoder(x)
        out = self.decoder(h)

        return out.view(B, C, H, W)


class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            # 28 x 28
            nn.Conv2d(in_channels, 4, kernel_size=5),
            # 4 x 24 x 24
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5),
            nn.ReLU(True),
            # 8 x 20 x 20 = 3200
            nn.Flatten(),
            nn.Linear(3200, latent_dim),
            # 10
            nn.Softmax(dim=latent_dim),
            )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(latent_dim, 400),
            # 400
            nn.ReLU(True),
            nn.Linear(400, 4000),
            # 4000
            nn.ReLU(True),
            nn.Unflatten(1, (10, 20, 20)),
            # 10 x 20 x 20
            nn.ConvTranspose2d(10, 10, kernel_size=5),
            # 24 x 24
            nn.ConvTranspose2d(10, 1, kernel_size=5),
            # 28 x 28
            nn.Sigmoid(),
            )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec


if __name__ == '__main__':
    pass
