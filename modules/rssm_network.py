import torch
import torch.nn as nn
from torch.distributions.utils import probs_to_logits
from .blocks import ResBlock, ImageChannelLayerNorm
from .utils import straight_through_categorical

######################## Encoder #########################

class Encoder(nn.Module):
    def __init__(self, config, codebook_weight=None):
        super().__init__()

        self.embed = nn.Embedding(config.vq_codebook_size, config.vq_code_dim)
        if codebook_weight is not None:
            self.embed.weight.data.copy_(codebook_weight)
            self.embed.weight.requires_grad = False 

        self.network = nn.Sequential(
            # (D, 16, 32) -> (64, 8, 16)
            nn.Conv2d(config.vq_code_dim, 64, 3, 2, 1, bias=False),
            ImageChannelLayerNorm(64),
            nn.SiLU(),

            # (64, 8, 16) -> (128, 4, 8)
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            ImageChannelLayerNorm(128),
            nn.SiLU(),
            ResBlock(128),

            # (128, 4, 8) -> (128, 2, 4)
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            ImageChannelLayerNorm(128),
            nn.SiLU(),

            nn.Flatten(),
            nn.Linear(128 * 2 * 4, config.encoded_state_size, bias=False),
            nn.LayerNorm(config.encoded_state_size),
            nn.SiLU(),
        )

    def forward(self, indices):
        if indices.ndim == 4:
            B, T, H, W = indices.shape
            x = indices.reshape(B * T, H, W)
            x = self.embed(x)
            x = x.permute(0, 3, 1, 2)
            x = self.network(x)
            x = x.view(B, T, -1)
        else:
            x = self.embed(indices)
            x = x.permute(0, 3, 1, 2)
            x = self.network(x)
        return x

######################## Decoder #########################

class Decoder(nn.Module):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(config.recurrent_size + config.latent_size, 128 * 2 * 4, bias=False),
            nn.LayerNorm(128 * 2 * 4, eps=eps),
            nn.SiLU(),
            nn.Unflatten(1, (128, 2, 4)),

            # (128, 2, 4) -> (96, 4, 8)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, 3, 1, 1, bias=False),
            ImageChannelLayerNorm(96),
            nn.SiLU(),

            # (96, 4, 8) -> (64, 8, 16)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, 3, 1, 1, bias=False),
            ImageChannelLayerNorm(64, eps),
            nn.SiLU(),

            # (64, 8, 16)-> (K, 16, 32)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            ImageChannelLayerNorm(64, eps),
            nn.SiLU(),
            nn.Conv2d(64, config.vq_codebook_size, 1)
        )

    def forward(self, hidden, latent):
        x = torch.cat((hidden, latent), dim=-1)

        if x.ndim == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            x = self.network(x)
            _, K, H, W = x.shape
            x = x.view(B, T, K, H, W)
        else:
            x = self.network(x)
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

    def get_logits(self, hidden):
        x = self.network(hidden)
        probs = x.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        uniform = torch.ones_like(probs) / self.config.latent_classes
        probs = (1 - self.config.uniform_mix) * probs + self.config.uniform_mix * uniform
        return probs_to_logits(probs)

    def forward(self, hidden):
        logits = self.get_logits(hidden)
        sample = straight_through_categorical(logits)
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

        sample = straight_through_categorical(logits)
        return sample.view(-1, self.config.latent_size), logits