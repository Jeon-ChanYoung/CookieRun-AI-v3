import torch.nn as nn

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

######################## Residual Block #########################

class ResBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = channels // reduction
        
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, hidden, 1),     
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), 
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),    
        )

    def forward(self, x):
        return x + self.block(x)

######################## SelfAttention #########################

"""
Norm SelfAttention
Reference: https://www.youtube.com/watch?v=U0s0f995w14

I originally intended to use it, 
but the model became too heavy, so I'm not using it here.
"""

# class SelfAttention(nn.Module):
#     def __init__(self, in_channels, heads=4):
#         super().__init__()

#         self.heads = heads
#         self.head_dim = in_channels // heads

#         # assert in_channels % heads == 0

#         self.norm  = nn.GroupNorm(8, in_channels)
#         self.query = nn.Conv2d(in_channels, in_channels, 1)
#         self.key   = nn.Conv2d(in_channels, in_channels, 1)
#         self.value = nn.Conv2d(in_channels, in_channels, 1)
#         self.out   = nn.Conv2d(in_channels, in_channels, 1)

#     def forward(self, x):
#         residual = x
#         B, C, H, W = x.shape

#         x = self.norm(x)
#         q = self.query(x) # (B, C, H, W)
#         k = self.key(x)   # (B, C, H, W)
#         v = self.value(x) # (B, C, H, W)

#         # Reshape to the number of heads
#         # (B, C, H, W) -> (B, heads, head_dim H*W)
#         q = q.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)
#         k = k.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)
#         v = v.view(B, self.heads, self.head_dim, H * W).permute(0, 1, 3, 2)

#         # # Attention
#         # # q shape: (B, heads, head_dim H*W)
#         # # k shape: (B, heads, head_dim H*W)
#         # # -------> (B, heads, H*W, H*W)
#         # attention = torch.einsum("bhdq,bhdk->bhqk", [q, k])
#         # attention = attention / (self.head_dim ** 0.5)
#         # attention = F.softmax(attention, dim=-1)

#         # # attention shape: (B, heads, H*W, H*W)
#         # #         v shape: (B, heads, head_dim H*W)
#         # # ---------------> (B, heads, head_dim H*W)
#         # out = torch.einsum("bhqk,bhdk->bhdq", [attention, v])
#         out = F.scaled_dot_product_attention(q, k, v)
#         out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
#         out = self.out(out)

#         return out + residual