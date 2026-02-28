import torch.nn as nn
import torch.nn.functional as F

######################## VAE-Encoder #########################

class VAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.GroupNorm(8, 64), 
            nn.SiLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            ResBlock(256),
            SelfAttention(256),

            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        self.mu_layer     = nn.Conv2d(256, config.latent_channel, 1)
        self.logvar_layer = nn.Conv2d(256, config.latent_channel, 1)

    def forward(self, x):
        x = self.network(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

######################## VAE-Decoder #########################

class VAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(config.latent_channel, 256, 3, 1, 1),
            ResBlock(256),
            SelfAttention(256),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GroupNorm(8, 128), 
            nn.SiLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.network(z)

######################## Residual Block #########################

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.block(x)

######################## SelfAttention #########################

"""
Norm SelfAttention
Reference: https://www.youtube.com/watch?v=U0s0f995w14
"""

class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super().__init__()

        self.heads = heads
        self.head_dim = in_channels // heads

        # assert in_channels % heads == 0

        self.norm  = nn.GroupNorm(8, in_channels)
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key   = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.out   = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        x = self.norm(x)
        q = self.query(x) # (B, C, H, W)
        k = self.key(x)   # (B, C, H, W)
        v = self.value(x) # (B, C, H, W)

        # Reshape to the number of heads
        # (B, C, H, W) -> (B, heads, head_dim H*W)
        q = q.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # # Attention
        # # q shape: (B, heads, head_dim H*W)
        # # k shape: (B, heads, head_dim H*W)
        # # -------> (B, heads, H*W, H*W)
        # attention = torch.einsum("bhdq,bhdk->bhqk", [q, k])
        # attention = attention / (self.head_dim ** 0.5)
        # attention = F.softmax(attention, dim=-1)

        # # attention shape: (B, heads, H*W, H*W)
        # #         v shape: (B, heads, head_dim H*W)
        # # ---------------> (B, heads, head_dim H*W)
        # out = torch.einsum("bhqk,bhdk->bhdq", [attention, v])
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.out(out)

        return out + residual