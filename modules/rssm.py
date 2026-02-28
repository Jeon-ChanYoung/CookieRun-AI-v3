import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal, Independent
from .rssm_network import Encoder, Decoder, RecurrentModel, TransitionModel, RepresentationModel

######################## Recurrent State Space Model #########################

class RSSM:
    def __init__(self, config):
        self.config = config
 
        # model
        self.encoder = Encoder().to(config.device)
        self.decoder = Decoder().to(config.device)
        self.recurrent_model = RecurrentModel().to(config.device)
        self.transition_model = TransitionModel().to(config.device)
        self.representation_model = RepresentationModel().to(config.device)

        # optimizer
        self.network_modules = [
            self.encoder,
            self.decoder,
            self.recurrent_model,
            self.transition_model,
            self.representation_model
        ]

        self.world_model_parameters = []
        for module in self.network_modules:
            self.world_model_parameters += list(module.parameters())

        self.world_model_optimizer = optim.AdamW(
            self.world_model_parameters,
            lr=config.world_model_lr,
            weight_decay=config.world_model_weight_decay
        )

        self.load_rssm(config.rssm_path)

    def train_step(self, states, actions):
        """
            states:  (B, T, 3, 128, 256)
            actions: (B, T, 3)
        """
        B, T = states.shape[:2] 
        encoded_states = self.encoder(states)

        hidden = torch.zeros(B, self.config.recurrent_size, device=self.config.device)
        latent = torch.zeros(B, self.config.latent_size, device=self.config.device)

        hiddens = torch.zeros(B, T-1, self.config.recurrent_size, device=self.config.device)
        latents = torch.zeros(B, T-1, self.config.latent_size, device=self.config.device)
        priors_logits = torch.zeros(B, T-1, self.config.latent_length, self.config.latent_classes, device=self.config.device)
        posteriors_logits = torch.zeros(B, T-1, self.config.latent_length, self.config.latent_classes, device=self.config.device)

        for t in range(1, T):
            hidden = self.recurrent_model(hidden, latent, actions[:, t-1])
            _, prior_logits = self.transition_model(hidden)
            latent, posterior_logits = self.representation_model(hidden, encoded_states[:, t])

            hiddens[:, t-1] = hidden
            latents[:, t-1] = latent
            priors_logits[:, t-1] = prior_logits
            posteriors_logits[:, t-1] = posterior_logits

        """
        hiddens:           (B, T-1, recurrent_size)
        latents:           (B, T-1, latent_length * latent_classes)
        priors_logits:     (B, T-1, latent_length, latent_classes)
        posteriors_logits: (B, T-1, latent_length, latent_classes)
        """

        ############# compute loss #############

        # recon loss
        reconstruction_means = self.decoder(hiddens, latents)
        reconstruction_dist = Independent(
            Normal(reconstruction_means, 1),
            len(self.config.observation_shape)
        )
        reconstruction_loss = -reconstruction_dist.log_prob(states[:, 1:]).mean()

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

        return reconstruction_loss.item(), kl_loss.item()
    

    def compute_kl(self, logits_p, logits_q):
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        return (p * (log_p - log_q)).sum(dim=(-2, -1))


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
        save_path = os.path.join(save_dir, f'oow_ep{epoch}.pth')
        torch.save({
            'encoder'               : self.encoder.state_dict(),
            'decoder'               : self.decoder.state_dict(),
            'recurrent_model'       : self.recurrent_model.state_dict(),
            'transition_model'      : self.transition_model.state_dict(),
            'representation_model'  : self.representation_model.state_dict(),
            'world_model_optimizer' : self.world_model_optimizer.state_dict()
        }, save_path)
        print(f"💾 RSSM Model saved: {save_path}")


    def load_rssm(self, check_point_path):
        print(f"📁 Loading checkpoint: {check_point_path}")
        checkpoint = torch.load(check_point_path, map_location=self.config.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.recurrent_model.load_state_dict(checkpoint['recurrent_model'])
        self.transition_model.load_state_dict(checkpoint['transition_model'])
        self.representation_model.load_state_dict(checkpoint['representation_model'])
        self.world_model_optimizer.load_state_dict(checkpoint['world_model_optimizer'])
        print("RSSM Checkpoint loaded successfully.")