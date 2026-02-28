import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits

######################## Encoder #########################

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1, bias=False),    # (3, 128, 256) -> (16, 64, 128)
            ImageChannelLayerNorm(16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1, bias=False),   # (16, 64, 128) -> (32, 32, 64)
            ImageChannelLayerNorm(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),   # (32, 32, 64)  -> (64, 16, 32)
            ImageChannelLayerNorm(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # (64, 16, 32)  -> (128, 8, 16)
            ImageChannelLayerNorm(128),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # (128, 8, 16)  -> (256, 4, 8)
            ImageChannelLayerNorm(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 4, 2, 1, bias=False), # (256, 4, 8)   -> (256, 2, 4)
            ImageChannelLayerNorm(256),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 2 * 4, self.config.encoded_state_size),
        )

    def forward(self, x):
        """
        x shape      : (B, T, 3, 128, 256)        or (B, 3, 128, 256)
        output shape : (B, T, encoded_state_size) or (B, encoded_state_size)
        """
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.network(x)
            x = x.view(B, T, -1)
        else:
            x = self.network(x)
        return x

######################## Decoder #########################

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size + self.config.latent_size, 256 * 2 * 4),
            nn.Unflatten(1, (256, 2, 4)),
            nn.ConvTranspose2d(256, 256, 4, 2, 1, bias=False), # (256, 2, 4) -> (256, 4, 8)
            ImageChannelLayerNorm(256),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # (256, 4, 8) -> (128, 8, 16)
            ImageChannelLayerNorm(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # (128, 8, 16) -> (64, 16, 32)
            ImageChannelLayerNorm(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),   # (64, 16, 32) -> (32, 32, 64)
            ImageChannelLayerNorm(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),   # (32, 32, 64) -> (16, 64, 128)
            ImageChannelLayerNorm(16),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),    # (16, 64, 128) -> (3, 128, 256)
        )

    def forward(self, hidden, latent):
        """
        x shape      : (B, recurrent_size + latent_size) or (B, T, recurrent_size + latent_size)
        output shape : (B, 3, 128, 256)                  or (B, T, 3, 128, 256)
        """
        x = torch.cat((hidden, latent), dim=-1)
        
        if x.ndim == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            x = self.network(x)
            x = x.view(B, T, *self.config.observation_shape)
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
    def __init__(self, config, hidden_size=256, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.latent_size + self.config.action_size, hidden_size, bias=False),
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
    def __init__(self, config, hidden_size=256, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size, hidden_size, bias=False),
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
    def __init__(self, config, hidden_size=256, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size + self.config.encoded_state_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),
            nn.Linear(hidden_size, self.config.latent_size),
        )

    def forward(self, hidden, latent):
        x = torch.cat((hidden, latent), -1)
        x = self.network(x)

        probs = x.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        uniform = torch.ones_like(probs) / self.config.latent_classes
        probs = (1 - self.config.uniform_mix) * probs + self.config.uniform_mix * uniform
        logits = probs_to_logits(probs)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.config.latent_size), logits