import torch
import torch.nn as nn

from torch.distributions.utils import probs_to_logits
from .blocks import ResBlock, DownBlock, UpBlock
from .utils import straight_through_categorical

######################## Encoder #########################

class Encoder(nn.Module):
    def __init__(self, config, codebook_weight=None):
        super().__init__()

        D = config.fsq_code_dim       # 3
        K = config.fsq_codebook_size  # 64
        E = config.encoded_state_size # 512

        if codebook_weight is None:
            raise TypeError("__init__() missing required argument: 'codebook_weight'")

        self.embed = nn.Embedding(K, D)
        self.embed.weight.data.copy_(codebook_weight)
        self.embed.weight.requires_grad = False

        self.network = nn.Sequential(
            nn.Conv2d(D, 64, 3, 1, 1), # (D, 8, 16) -> (64, 8, 16)
            nn.GroupNorm(32, 64),
            nn.SiLU(),

            DownBlock(64, 128),  # (D=3, 8, 16) -> (64, 4, 8)
            DownBlock(128, 256), # (64, 4, 8)   -> (128, 2, 4)

            nn.Flatten(),
            nn.Linear(256 * 2 * 4, E, bias=False),
            nn.LayerNorm(E),
            nn.SiLU()
        )

    def forward(self, indices):
        """
            indices: (B, 8, 16)              or (B, T, 8, 16)
            output:  (B, encoded_state_size) or (B, T, encoded_state_size)
        """
        if indices.ndim == 4:
            B, T, H, W = indices.shape
            x = self.embed(indices.reshape(B * T, H, W)) # (BT, H, W, D)
            x = x.permute(0, 3, 1, 2)                    # (BT, D, H, W)
            x = self.network(x)
            return x.view(B, T, -1)
        else:
            x = self.embed(indices)   # (B, H, W, D)
            x = x.permute(0, 3, 1, 2) # (B, D, H, W)
            return self.network(x)

######################## Decoder #########################

class Decoder(nn.Module):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        
        K = config.fsq_codebook_size # 64
        R = config.recurrent_size    # 512
        L = config.latent_size       # 1024

        self.network = nn.Sequential(
            nn.Linear(R + L, 256 * 2 * 4, bias=False),
            nn.LayerNorm(256 * 2 * 4, eps=eps),
            nn.SiLU(),
            nn.Unflatten(1, (256, 2, 4)),

            UpBlock(256, 128), # (256, 2, 4) -> (128, 4, 8)
            UpBlock(128, 64),  # (128, 4, 8) -> (64, 8, 16)

            nn.Conv2d(64, K, 1),  # (64, 8, 16) -> (K, 8, 16)
        )

    def forward(self, hidden, latent):
        x = torch.cat((hidden, latent), dim=-1)
        if x.ndim == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            x = self.network(x)
            _, K, H, W = x.shape
            return x.view(B, T, K, H, W)
        else:
            return self.network(x)
    
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

        R = config.recurrent_size # 512
        L = config.latent_size    # 1024
        A = config.action_size    # 3
        
        self.network = nn.Sequential(
            nn.Linear(R + A, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),
            
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),
        )
        
        self.recurrent = GRUCell(hidden_size, R)

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

        R = config.recurrent_size # 512
        L = config.latent_size    # 1024

        self.network = nn.Sequential(
            nn.Linear(R, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, L),
        )

    def get_logits(self, hidden):
        x = self.network(hidden)
        probs = x.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        probs = (1 - self.config.uniform_mix) * probs + self.config.uniform_mix / self.config.latent_classes
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

        E = config.encoded_state_size # 512
        R = config.recurrent_size     # 512
        L = config.latent_size        # 1024

        self.network = nn.Sequential(
            nn.Linear(R + E, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.LayerNorm(hidden_size, eps=eps),
            nn.SiLU(),

            nn.Linear(hidden_size, L),
        )

    def forward(self, hidden, encoded_state):
        x = torch.cat((hidden, encoded_state), -1)
        x = self.network(x)

        probs = x.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        probs = (1 - self.config.uniform_mix) * probs + self.config.uniform_mix / self.config.latent_classes
        logits = probs_to_logits(probs)

        sample = straight_through_categorical(logits)
        return sample.view(-1, self.config.latent_size), logits