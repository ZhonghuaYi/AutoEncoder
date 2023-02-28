import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input):
        raise NotImplementedError

    def decode(self, input):
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs):
        raise NotImplementedError

    def generate(self, x, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs):
        pass

    @abstractmethod
    def loss_function(self, *inputs, **kwargs):
        pass


class VAE(BaseVAE):

    def __init__(self, in_channels, latent_dim, hidden_dims=None, **kwargs):
        super().__init__()

        self.latent_dim = latent_dim
        self._hidden_dims = hidden_dims.copy()
        self._in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_pooling = nn.AdaptiveAvgPool2d((2, 2))

        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # Build decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], kernel_size=3, stride=2, padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=self._in_channels, kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, input):
        result = self.encoder(input)
        result = self.encoder_pooling(result)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        var = self.fc_var(result)

        return [mu, var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self._hidden_dims[-1], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        if recons.shape[-1] != input.shape[-1] or recons.shape[-2] != recons.shape[-2]:
            width, height = input.shape[-1], input.shape[-2]
            recons = F.interpolate(recons, (height, width), mode='bilinear', align_corners=False)

        return [recons, input, mu, log_var]

    def sample(self, num_samples, current_device, **kwargs):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)

        return samples

    def generate(self, x, **kwargs):
        return self.forward(x)[0]
