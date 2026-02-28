import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .rssm_network import LatentEncoder, LatentDecoder, RecurrentModel, TransitionModel, RepresentationModel

######################## Recurrent State Space Model #########################

class RSSM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
 
        # model
        self.encoder = LatentEncoder(config)
        self.decoder = LatentDecoder(config)
        self.recurrent_model = RecurrentModel(config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

        # optimizer
        self.rssm_optimizer = optim.Adam(self.parameters(), lr=config.rssm_lr)

        # normalization
        self.register_buffer('z_mean', torch.zeros(4, 1, 1))
        self.register_buffer('z_std',  torch.ones(4, 1, 1))

        self.load_rssm(config.rssm_path)

    def train_step(self, zs, actions):
        """
            zs:      (B, T, 4, 16, 32)
            actions: (B, T, 3)
        """
        B, T = zs.shape[:2] 
        zs_norm = self.normalize(zs)
        encoded_states = self.encoder(zs_norm)

        hidden = torch.zeros(B, self.config.recurrent_size, device=self.config.device)
        latent = torch.zeros(B, self.config.latent_size, device=self.config.device)

        hiddens = []                                     
        latents = []
        priors_logits = []
        posteriors_logits = []

        for t in range(1, T):
            hidden = self.recurrent_model(hidden, latent, actions[:, t-1])
            _, prior_logits = self.transition_model(hidden)
            latent, posterior_logits = self.representation_model(hidden, encoded_states[:, t])

            hiddens.append(hidden)
            latents.append(latent)
            priors_logits.append(prior_logits)
            posteriors_logits.append(posterior_logits)

        hiddens = torch.stack(hiddens, dim=1)            
        latents = torch.stack(latents, dim=1)            
        priors_logits = torch.stack(priors_logits, dim=1)
        posteriors_logits = torch.stack(posteriors_logits, dim=1)

        """
        hiddens:           (B, T-1, recurrent_size)
        latents:           (B, T-1, latent_length * latent_classes)
        priors_logits:     (B, T-1, latent_length, latent_classes)
        posteriors_logits: (B, T-1, latent_length, latent_classes)
        """

        ############# compute loss #############

        # recon loss
        predicted_latents = self.decoder(hiddens, latents)
        target_latents = zs_norm[:, 1:]
        reconstruction_loss = F.mse_loss(predicted_latents, target_latents)

        # kl loss
        prior_loss = self.compute_kl(posteriors_logits.detach(), priors_logits)
        prior_loss = self.config.prior_coefficient * prior_loss.clamp(min=self.config.free_nat)
        posterior_loss = self.compute_kl(posteriors_logits, priors_logits.detach())
        posterior_loss = self.config.posterior_coefficient * posterior_loss.clamp(min=self.config.free_nat)
        kl_loss = (prior_loss + posterior_loss).mean()

        loss = reconstruction_loss + kl_loss

        ############# backprop #############

        self.world_model_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.world_model_parameters, 
            self.config.gradient_clip, 
            self.config.gradient_norm_type
        )
        self.world_model_optimizer.step()

        return loss.item(), reconstruction_loss.item(), kl_loss.item()
    

    def compute_kl(self, logits_p, logits_q):
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        return (p * (log_p - log_q)).sum(dim=(-2, -1))
    

    def normalize(self, z):
        return (z - self.z_mean) / self.z_std


    def denormalize(self, z_norm):
        return z_norm * self.z_std + self.z_mean


    def set_latent_stats(self, mean, std):
        self.z_mean.copy_(mean)
        self.z_std.copy_(std)
        print(f"Latent stats set: mean={mean.squeeze()}, std={std.squeeze()}")


    def change_train_mode(self, train=True):
        if not train:
            for module in self.network_modules:
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()
        else:
            for module in self.network_modules:
                for param in module.parameters():
                    param.requires_grad = True
                module.train()


    def save_rssm(self, epoch, save_dir):
        save_path = os.path.join(save_dir, f'rssm_ep{epoch}.pth')
        torch.save({
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrent_model'       : self.recurrent_model.state_dict(),
            'transition_model'      : self.transition_model.state_dict(),
            'representation_model'  : self.representation_model.state_dict(),
            'rssm_optimizer'        : self.rssm_optimizer.state_dict(),
            'z_mean'                : self.z_mean,
            'z_std'                 : self.z_std,
        }, save_path)
        print(f"RSSM Model saved: {save_path}")


    def load_rssm(self, check_point_path):
        print(f"Loading checkpoint: {check_point_path}")
        checkpoint = torch.load(check_point_path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrent_model.load_state_dict(checkpoint['recurrent_model'])
        self.transition_model.load_state_dict(checkpoint['transition_model'])
        self.representation_model.load_state_dict(checkpoint['representation_model'])
        self.rssm_optimizer.load_state_dict(checkpoint['rssm_optimizer'])
        self.z_mean.copy_(checkpoint['z_mean'])
        self.z_std.copy_(checkpoint['z_std'])
        print("RSSM Checkpoint loaded successfully.")