import torch.nn as nn

from .blocks import ResBlock, SelfAttention

######################## VAE-Encoder #########################

class VAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),                   
            ResBlock(64),  

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock(128),                  
            ResBlock(128),   

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            ResBlock(256),
            ResBlock(256),                  
            SelfAttention(256),
            ResBlock(256),         

            nn.GroupNorm(8, 256),
            nn.SiLU()
        )
        self.mu_layer     = nn.Conv2d(256, config.latent_channel, 1)
        self.logvar_layer = nn.Conv2d(256, config.latent_channel, 1)

    def forward(self, x):
        h = self.network(x)
        return self.mu_layer(h), self.logvar_layer(h)

######################## VAE-Decoder #########################

class VAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(config.latent_channel, 256, 3, 1, 1),
            ResBlock(256),
            ResBlock(256),                  
            SelfAttention(256),
            ResBlock(256), 

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock(128),                  
            ResBlock(128),                  

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),                  
            ResBlock(64),   

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.network(z)