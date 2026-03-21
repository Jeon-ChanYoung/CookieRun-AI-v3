import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .vqvae_network import (
    VQVAEEncoder,
    VQVAEDecoder,
    FiniteScalarQuantizer,
    VGGPerceptualLoss,
)

######################## (FSQ) Vector Quantized VAE #########################


class VQVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # model
        self.encoder = VQVAEEncoder(config)
        self.quantizer = FiniteScalarQuantizer(config)
        self.decoder = VQVAEDecoder(config)
        self.perceptual = VGGPerceptualLoss()

        # optimizer
        self.vqvae_params = list(self.encoder.parameters()) + list(
            self.decoder.parameters()
        )
        self.optimizer = optim.Adam(
            self.vqvae_params, lr=config.vqvae_lr
        )  # vqvae_lr = 0.001


    def forward(self, x):
        z_e = self.encoder(x)  # (B, D, 8, 16)
        z_q, indices = self.quantizer(z_e)  
        recon = self.decoder(z_q)  # (B, 3, 128, 256)
        return recon, indices
        

    @torch.no_grad()
    def encode(self, x):
        z_e = self.encoder(x)
        _, indices = self.quantizer(z_e)
        return indices  # (B, 8, 16)

    
    @torch.no_grad()
    def decode(self, indices):
        z_q = self.quantizer.get_codebook_entry(indices)  # (B, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2).float()  # (B, D, H, W)
        return self.decoder(z_q)
        

    def train_step(self, x):
        recon, indices = self(x)
        recon_loss = 0.5 * F.mse_loss(recon, x) + F.l1_loss(recon, x)
        p_loss = self.perceptual(recon.float(), x.float())

        loss = recon_loss + p_loss * self.config.perceptual_weight

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.vqvae_params, max_norm=5.0)
        self.optimizer.step()

        return (
            loss.item(),
            recon_loss.item(),
            p_loss.item(),
            self.quantizer.usage,
        )
        

    def change_train_mode(self, train=True):
        if train:
            self.train()
            for p in self.parameters():
                p.requires_grad = True

            self.perceptual.eval()
            for p in self.perceptual.parameters():
                p.requires_grad = False
        else:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False


    # def visualize_recon(self, frame_loader, n=8):
    #     self.change_train_mode(train=False)
    #     originals = next(iter(frame_loader))[:n].to(self.config.device)
    #     indices = self.encode(originals)
    #     recons = self.decode(indices)
    #     fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
        
    #     for i in range(n):
    #         axes[0, i].imshow(originals[i].cpu().permute(1, 2, 0).numpy())
    #         axes[0, i].axis('off')
    #         axes[1, i].imshow(recons[i].cpu().clamp(0, 1).permute(1, 2, 0).numpy())
    #         axes[1, i].axis('off')
            
    #     axes[0, 0].set_title("Original"); axes[1, 0].set_title("FSQ Recon")
    #     plt.suptitle(f"FSQ levels={self.config.fsq_levels}, K={self.config.fsq_codebook_size}")
    #     plt.tight_layout()
    #     plt.show()
    #     plt.close()


    def save_vqvae(self, epoch, save_dir):
        save_path = os.path.join(save_dir, f'vqvae_ep{epoch}.pth')

        torch.save({
            'encoder':   self.encoder.state_dict(),
            'decoder':   self.decoder.state_dict(),
            'quantizer': {k: v for k, v in self.quantizer.state_dict().items()},
            'optimizer': self.optimizer.state_dict()
        }, save_path)
        print(f"VQ-VAE saved: {save_path}")
        

    def load_vqvae(self, checkpoint_path):
        print(f"Loading VQ-VAE: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.quantizer.load_state_dict(checkpoint['quantizer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("VQ-VAE Checkpoint loaded successfully.")
