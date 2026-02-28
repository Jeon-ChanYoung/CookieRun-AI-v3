import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

######################## Latent Flow Matching(Decoder) #########################

class LatentFlowDecoder(nn.Module):
    def __init__(
        self,
        config,
        base_channels=64,
        channel_mult=(1, 2),
        cond_dim=128,
        rssm_dim=768,
        spatial_channels=16
    ):
        super().__init__()

        in_channels = config.latent_channel
        rssm_dim = config.recurrent_size + config.latent_size
        channels = [base_channels * m for m in channel_mult]  # [64, 128]

        self.time_layer = nn.Sequential(
            SinusoidalPositionalEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.cond_layer = nn.Sequential(
            nn.Linear(rssm_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.spatial_projector = nn.Sequential(
            nn.Linear(rssm_dim, spatial_channels * 2),
            nn.SiLU(),
            nn.Linear(spatial_channels * 2, spatial_channels),
        )

        self.in_conv = nn.Conv2d(in_channels + spatial_channels, channels[0], 3, 1, 1)

        self.downs = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.downs.append(LFMResBlock(channels[i], channels[i], cond_dim))
            self.down_pools.append(nn.Conv2d(channels[i], channels[i + 1], 4, 2, 1))

        self.mid = LFMResBlock(channels[-1], channels[-1], cond_dim)

        self.up_samples = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i in reversed(range(len(channels) - 1)):
            self.up_samples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels[i + 1], channels[i], 3, 1, 1),
            ))
            self.ups.append(LFMResBlock(channels[i] * 2, channels[i], cond_dim))

        self.out = nn.Sequential(
            nn.GroupNorm(8, channels[0]), nn.SiLU(),
            nn.Conv2d(channels[0], in_channels, 3, 1, 1),
        )  


    def forward(self, z_t, t, rssm_state):
        c = self.time_layer(t) + self.cond_layer(rssm_state)

        spatial = self.spatial_projector(rssm_state)
        spatial = spatial[..., None, None].expand(-1, -1, z_t.shape[2], z_t.shape[3])
        z = torch.cat([z_t, spatial], dim=1)
        z = self.in_conv(z)

        skip_connections = []
        for res_block, downsample in zip(self.downs, self.down_pools):
            z = res_block(z, c)
            skip_connections.append(z)
            z = downsample(z)

        z = self.mid(z, c)

        for upsample, up_block, skip in zip(self.up_samples, self.ups, reversed(skip_connections)):
            z = upsample(z)
            z = torch.cat([z, skip], dim=1)
            z = up_block(z, c)

        return self.out(z)

######################## SinusoidalPositionalEmbedding #########################

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device).float() * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
    
######################## LFMResBlock #########################

class LFMResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.agn1  = AdaptiveGroupNorm(in_channels, cond_dim)
        self.conv1 = DSConv(in_channels, out_channels)
        self.agn2  = AdaptiveGroupNorm(out_channels, cond_dim)
        self.conv2 = DSConv(out_channels, out_channels)
        self.skip  = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.agn1(x, cond)))
        h = self.conv2(F.silu(self.agn2(h, cond)))
        return h + self.skip(x)
    
######################## AdaptiveGroupNorm #########################

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels, affine=False)
        self.cond_to_affine = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        h = self.norm(x)
        scale, shift = self.cond_to_affine(cond).chunk(2, dim=-1) # [B, C], [B, C]
        scale = scale[..., None, None]  # [B, C, 1, 1]
        shift = shift[..., None, None]  # [B, C, 1, 1]
        return h * (1.0 + scale) + shift

######################## Depthwise Separable Conv #########################

class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.pw(self.dw(x))
    
######################## Exponential Moving Average #########################

class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.step = 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.lerp_(p, 1 - d)