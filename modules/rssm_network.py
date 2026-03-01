import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits
from .blocks import ResBlock, SelfAttention

######################## LatentEncoder #########################

class LatentEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.network = nn.Sequential(
            # (4, 16, 32) -> (64, 8, 16)
            nn.Conv2d(4, 64, 3, 2, 1, bias=False),
            ImageChannelLayerNorm(64),
            nn.SiLU(),

            # (64, 8, 16) -> (128, 4, 8)
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            ImageChannelLayerNorm(128),
            nn.SiLU(),

            ResBlock(128),
            SelfAttention(128),

            # (128, 4, 8) -> (256, 2, 4)
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            ImageChannelLayerNorm(256),
            nn.SiLU(),

            ResBlock(256),

            nn.Flatten(),
            nn.Linear(256 * 2 * 4, config.encoded_state_size, bias=False),
            nn.LayerNorm(config.encoded_state_size),
            nn.SiLU(),
        )

    def forward(self, x):
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.network(x)
            x = x.view(B, T, -1)
        else:
            x = self.network(x)
        return x

######################## LatentDecoder #########################

class LatentDecoder(nn.Module):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.recurrent_size + config.latent_size, 256 * 2 * 4, bias=False),
            nn.LayerNorm(256 * 2 * 4, eps=eps),
            nn.SiLU(),
            nn.Unflatten(1, (256, 2, 4)),

            ResBlock(256),

            # (256, 2, 4) -> (128, 4, 8)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            ImageChannelLayerNorm(128),
            nn.SiLU(),

            ResBlock(128),
            SelfAttention(128),

            # (128, 4, 8) -> (64, 8, 16)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            ImageChannelLayerNorm(64, eps),
            nn.SiLU(),

            # (64, 8, 16) -> (4, 16, 32)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, config.latent_channel, 3, 1, 1),
        )

    def forward(self, hidden, latent):
        x = torch.cat((hidden, latent), dim=-1)

        if x.ndim == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            x = self.network(x)
            x = x.view(B, T, *self.config.latent_shape_vae)
        else:
            x = self.network(x)
        return x

######################## ImageChannelLayerNorm #########################

class ImageChannelLayerNorm(nn.Module):
    def __init__(self, in_channels, eps=1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x
    
######################## RecurrentModel #########################

"""
Custom GRUCell
Reference: https://github.com/danijar/dreamerv3
"""

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, update_bias=-1, eps=1e-3):
        super().__init__()
        self.update_bias = update_bias
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=False)
        self.norm = nn.LayerNorm(3 * hidden_size, eps=eps)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        parts = self.norm(self.linear(combined))

        reset, candidate, update = torch.chunk(parts, 3, dim=-1)

        reset     = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update    = torch.sigmoid(update + self.update_bias)

        out = update * candidate + (1 - update) * h
        return out

class RecurrentModel(nn.Module):
    def __init__(self, config, hidden_size=512, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.latent_size + self.config.action_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),
        )
        self.recurrent = GRUCell(hidden_size, self.config.recurrent_size)

    def forward(self, hidden, latent, action):
        x = torch.cat((latent, action), -1)
        x = self.network(x)
        x = self.recurrent(x, hidden)
        return x
    
######################## TransitionModel #########################

class TransitionModel(nn.Module):
    def __init__(self, config, hidden_size=512, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, self.config.latent_size),
        )

    def forward(self, hidden):
        x = self.network(hidden)

        probs = x.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        uniform = torch.ones_like(probs) / self.config.latent_classes
        probs = (1 - self.config.uniform_mix) * probs + self.config.uniform_mix * uniform
        logits = probs_to_logits(probs)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.config.latent_size), logits
    
######################## RepresentationModel #########################

class RepresentationModel(nn.Module):
    def __init__(self, config, hidden_size=512, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size + self.config.encoded_state_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, self.config.latent_size),
        )

    def forward(self, hidden, encoded_state):
        x = torch.cat((hidden, encoded_state), -1)
        x = self.network(x)

        probs = x.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        uniform = torch.ones_like(probs) / self.config.latent_classes
        probs = (1 - self.config.uniform_mix) * probs + self.config.uniform_mix * uniform
        logits = probs_to_logits(probs)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.config.latent_size), logits