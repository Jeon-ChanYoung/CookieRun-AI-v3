import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .vae_network import VAEEncoder, VAEDecoder

######################## Variational AutoEncoder #########################

class VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # model
        self.encoder = VAEEncoder(config)
        self.decoder = VAEDecoder(config)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config.vae_lr) # vae_lr = 0.0003
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, config.vae_train_epochs, eta_min=0.00001
        ) # vae_train_epochs = 30

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10, max=5)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


    @torch.no_grad()
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu


    @torch.no_grad()
    def decode(self, z):
        return self.decoder(z)


    def train_step(self, x, epoch):
        recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(recon, x) + F.l1_loss(recon, x) * 0.3
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weight = min(1.0, epoch / 20) * self.config.vae_kl_weight # vae_kl_weight = 0.000001

        loss = recon_loss + kl_loss * kl_weight

        self.optimizer.zero_grad()    
        loss.backward()           
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()               

        return loss.item(), recon_loss.item(), kl_loss.item()


    def change_train_mode(self, train=True):
        if train:
            self.train()
            for p in self.parameters():
                p.requires_grad = True
        else:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False


    def step_scheduler(self):
        self.scheduler.step()


    def save_vae(self, epoch, save_dir):
        save_path = os.path.join(save_dir, f'vae_ep{epoch}.pth')
        torch.save({
            'encoder' : self.encoder.state_dict(),
            'decoder' : self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)
        print(f"VAE Model saved: {save_path}")


    def load_vae(self, checkpoint_path):
        print(f"Loading VAE: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        print("VAE Checkpoint loaded successfully.")