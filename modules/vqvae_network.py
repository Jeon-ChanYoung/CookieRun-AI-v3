import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock, DownBlock, UpBlock

######################## VQ-VAE-Encoder #########################

class VQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        D = config.fsq_code_dim # 3

        self.network = nn.Sequential(
            DownBlock(3, 64),    # (B, 3, 128, 256) -> (B, 64, 64, 128)
            ResBlock(64),

            DownBlock(64, 128),  # (B, 64, 64, 128) -> (B, 128, 32, 64)
            ResBlock(128),
            ResBlock(128),

            DownBlock(128, 256), # (B, 128, 32, 64) -> (B, 256, 16, 32)
            ResBlock(256),
            ResBlock(256),

            DownBlock(256, 256), # (B, 256, 16, 32) -> (B, 256, 8, 16)
            ResBlock(256),
            ResBlock(256),

            nn.Conv2d(256, D, 1)
        )

    def forward(self, x):
        return self.network(x)  # (B, D, 8, 16)

######################## VQ-VAE-Decoder #########################

class VQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
    
        D = config.fsq_code_dim # 3       
        
        self.network = nn.Sequential(
            nn.Conv2d(D, 256, 3, 1, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),

            ResBlock(256),
            ResBlock(256),
            UpBlock(256, 256),         # (B, 256, 8, 16) -> (B, 256, 16, 32)

            ResBlock(256),
            ResBlock(256),
            UpBlock(256, 128),         # (B, 256, 16, 32) -> (B, 128, 32, 64)

            ResBlock(128),
            ResBlock(128),
            UpBlock(128, 64),          # (B, 128, 32, 64) -> (B, 64, 64, 128)

            ResBlock(64),
            UpBlock(64, 3, last=True), # (B, 64, 64, 128) -> (B, 3, 128, 256)
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.network(z)
    
######################## FiniteScalarQuantizer ########################

class FiniteScalarQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.D = config.fsq_code_dim      # D = 3
        self.K = config.fsq_codebook_size # K = 64

        _levels = torch.tensor(config.fsq_levels, dtype=torch.float32)
        self.register_buffer('_levels', _levels)

        basis = []
        stride = 1    
        for level in reversed(config.fsq_levels): # [4, 4, 4] -> [4, 4, 4]
            basis.insert(0, stride)
            stride *= level

        # basis = [16, 4, 1]
        self.register_buffer('_basis', torch.tensor(basis, dtype=torch.int64))

        codebook = self._build_codebook()
        self.register_buffer('codebook', codebook)

    
    def _build_codebook(self):
        grids = [torch.arange(level) for level in self._levels.long().tolist()]
        # grids = [tensor([0, 1, 2, 3]), 
        #          tensor([0, 1, 2, 3]),
        #          tensor([0, 1, 2, 3])]

        # make all 3D coordinates combinations (4x4x4=64)
        mesh = torch.meshgrid(*grids, indexing='ij')

        codes = torch.stack([m.reshape(-1) for m in mesh], dim=-1).float()
        # codes.shape = (64, 3)
        # codes[0]  = tensor([0., 0., 0.])
        # codes[1]  = tensor([0., 0., 1.])
        # ...
        # codes[62] = tensor([3., 3., 2.])
        # codes[63] = tensor([3., 3., 3.])

        # from [0, 3] to [-1, 1]
        half = (self._levels - 1) / 2  # tensor([1.5, 1.5, 1.5])
        codes = (codes - half) / half # normalize to [-1, 1]
        # codes.shape = (64, 3)
        # codes[0]  = tensor([-1., -1., -1.])
        # codes[1]  = tensor([-1., -1., -0.3333])
        # ...
        # codes[42] = tensor([ 0.3333, -0.3333,  0.3333])
        # codes[63] = tensor([1., 1., 1.])
        
        return codes


    def forward(self, z_e):
        # z_e.shape = (B, D, H, W) = (B, 3, 8, 16)

        z = z_e.permute(0, 2, 3, 1) # (B, 8, 16, 3)

        z_q, z_bounded = self._quantize(z)
        # z_q.shape = (B, 8, 16, 3): grid value
        # z_bounded.shape = (B, 8, 16, 3): for gradient

        indices = self._codes_to_indices(z_q)
        # indices.shape = (B, 8, 16): index of codebook

        # STE
        z_q_st = z_bounded + (z_q - z_bounded).detach()
        z_q_st = z_q_st.permute(0, 3, 1, 2) # (B, 3, 8, 16)

        return z_q_st, indices


    def _quantize(self, z):
        # z.shape = (B, 8, 16, 3)
        # ex) let’s consider a single pixel vector: z = tensor([0.5, -0.2, 2.0]).

        half = (self._levels - 1) / 2  # tensor([1.5, 1.5, 1.5])
        
        z_01 = torch.sigmoid(z) 
        # 0 ~ 1 scaling
        # z_01 = tensor([0.622, 0.447, 0.880])

        z_scaled = z_01 * (self._levels - 1) 
        # 0 ~ 3 scaling
        # z_scaled = tensor([1.866, 1.350, 2.640])])

        z_rounded = torch.round(z_scaled)
        # z_rounded = tensor([2., 1., 3.])

        z_q = z_rounded / half - 1
        # z_q = tensor([0.333, -0.333, 1.])

        z_bounded = z_01 * 2 - 1
        # -1 ~ 1 scaling / for gradient
        # z_bounded = tensor([0.244, -0.100, 0.760])

        return z_q, z_bounded


    def _codes_to_indices(self, z_q):
        # z_q.shape = (B, 8, 16, 3)
        # ex) z_q = tensor([0.333, -0.333, 1.])
        half = (self._levels - 1) / 2  # tensor([1.5, 1.5, 1.5])

        level_indices = torch.round(z_q * half + half.long())
        # level_indices = tensor([2., 1., 3.])

        indices = (level_indices * self._basis).sum(dim=-1)
        # indices = tensor([2*16 + 1*4 + 3*1]) = tensor(39)

        return indices


    def get_codebook_entry(self, indices):
        # indices.shape = (B, 8, 16)
        return F.embedding(indices.long(), self.codebook)

    @property
    def usage(self):
        return self.K

#################### VGGPerceptualLoss ####################

"""
If you want to train this repo, use it.
pip install torchvision
from torchvision.models import vgg16, VGG16_Weights
...

"""
class VGGPerceptualLoss(nn.Module):
    pass

# class VGGPerceptualLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

#         self.blocks = nn.ModuleList([vgg[:4], vgg[4:9]])

#         for p in self.parameters():
#             p.requires_grad = False

#         self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
#         self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

#     def train(self, mode=True):
#         return super().train(False)

#     def forward(self, pred, target):
#         pred   = (pred   - self.mean) / self.std
#         target = (target - self.mean) / self.std

#         loss = 0.0
#         x, y = pred, target
#         for block in self.blocks:
#             x = block(x)

#             with torch.no_grad():
#                 y = block(y)

#             loss += F.mse_loss(x, y)
#         return loss