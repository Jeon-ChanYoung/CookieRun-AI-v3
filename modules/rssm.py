import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .rssm_network import Encoder, Decoder, RecurrentModel, TransitionModel, RepresentationModel

######################## Recurrent State Space Model #########################

class RSSM(nn.Module):
    def __init__(self, config, codebook_weight=None):
        super().__init__()

        self.config = config
 
        # model
        self.encoder = Encoder(config, codebook_weight)
        self.decoder = Decoder(config)
        self.recurrent_model = RecurrentModel(config)
        self.transition_model = TransitionModel(config)
        self.representation_model = RepresentationModel(config)

        # optimizer
        self.rssm_optimizer = optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=config.rssm_lr # rssm_lr = 0.0003
        )


    def train_step(self, indices, actions):
        B, T = indices.shape[:2] 
        encoded_states = self.encoder(indices)

        hidden = torch.zeros(B, self.config.recurrent_size, device=self.config.device)
        latent, _ = self.representation_model(hidden, encoded_states[:, 0])

        hiddens = []
        latents = []
        posteriors_logits = []

        for t in range(1, T):
            hidden = self.recurrent_model(hidden, latent, actions[:, t-1])
            latent, posterior_logits = self.representation_model(
                hidden, encoded_states[:, t]
            )
            hiddens.append(hidden)
            latents.append(latent)
            posteriors_logits.append(posterior_logits)

        hiddens = torch.stack(hiddens, dim=1)
        latents = torch.stack(latents, dim=1)
        posteriors_logits = torch.stack(posteriors_logits, dim=1)

        priors_logits = self.transition_model.get_logits(
            hiddens.view(B * (T-1), self.config.recurrent_size)
        ).view(B, T-1, self.config.latent_length, self.config.latent_classes)

        """
        hiddens:           (B, T-1, recurrent_size)
        latents:           (B, T-1, latent_length * latent_classes)
        priors_logits:     (B, T-1, latent_length, latent_classes)
        posteriors_logits: (B, T-1, latent_length, latent_classes)
        """

        ############# compute loss #############

        # recon loss
        predicted_logits = self.decoder(hiddens, latents) # (B, T-1, K=256, 16, 32)
        target_indices = indices[:, 1:]  # (B, T-1, 16, 32)

        pred_flat = predicted_logits.permute(0, 1, 3, 4, 2).reshape(-1, self.config.vq_codebook_size)
        target_flat = target_indices.reshape(-1)
        reconstruction_loss = F.cross_entropy(pred_flat, target_flat, label_smoothing=self.config.label_smoothing) # label_smoothing = 0.1

        # for logging
        with torch.no_grad():
            accuracy = (pred_flat.argmax(-1) == target_flat).float().mean()

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

        return loss.item(), reconstruction_loss.item(), kl_loss.item(), accuracy.item()
    

    def compute_kl(self, logits_p, logits_q):
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        return (p * (log_p - log_q)).sum(dim=(-2, -1))
    

    def change_train_mode(self, train=True):
        if train:
            self.train()
            for name, p in self.named_parameters():
                p.requires_grad = 'embed.weight' not in name
        else:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False


    @torch.no_grad()
    def logits_to_indices(self, logits):
        # logits: (..., K, H, W) -> indices: (..., H, W)
        return logits.argmax(dim=-3)
    

    # @torch.no_grad()
    # def visualize(self, vqvae, rssm_loader, epoch=0, n_frames=10, save_dir=None):


    def save_rssm(self, epoch, save_dir):
        def strip_prefix(state_dict):
            return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        save_path = os.path.join(save_dir, f'rssm_ep{epoch}.pth')
        torch.save({
            'encoder'               : strip_prefix(self.encoder.state_dict()),
            'decoder'               : strip_prefix(self.decoder.state_dict()),
            'recurrent_model'       : strip_prefix(self.recurrent_model.state_dict()),
            'transition_model'      : strip_prefix(self.transition_model.state_dict()),
            'representation_model'  : strip_prefix(self.representation_model.state_dict()),
            'rssm_optimizer'        : self.rssm_optimizer.state_dict()
        }, save_path)
        print(f"RSSM Model saved: {save_path}")


    def load_rssm(self, check_point_path):
        print(f"Loading checkpoint: {check_point_path}")
        checkpoint = torch.load(check_point_path, map_location=self.config.device)

        def strip_prefix(state_dict):
            return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.encoder.load_state_dict(strip_prefix(checkpoint['encoder']))
        self.decoder.load_state_dict(strip_prefix(checkpoint['decoder']))
        self.recurrent_model.load_state_dict(strip_prefix(checkpoint['recurrent_model']))
        self.transition_model.load_state_dict(strip_prefix(checkpoint['transition_model']))
        self.representation_model.load_state_dict(strip_prefix(checkpoint['representation_model']))
        self.rssm_optimizer.load_state_dict(checkpoint['rssm_optimizer'])
        print("RSSM Checkpoint loaded successfully.")