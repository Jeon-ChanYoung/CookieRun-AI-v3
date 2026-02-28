import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .latent_flow_matching import LatentFlowDecoder, ExponentialMovingAverage

######################## LFM World Model #########################

class LFMWorldModel:
    def __init__(self, config, vae, rssm):
        self.config = config
        self.vae = vae
        self.rssm = rssm

        # eval
        self.vae.change_train_mode(train=False)
        self.rssm.change_train_mode(train=False)

        # model
        self.flow_decoder = LatentFlowDecoder().to(config)
        self.ema = ExponentialMovingAverage(self.flow_decoder, decay=config.ema_decay)
        self.scale = 1.0

        # optimizer
        self.world_model_parameters = list(self.flow_decoder.parameters())
        self.optimizer = optim.AdamW(
            self.world_model_parameters,
            lr=self.config.flow_lr,
            weight_decay=self.config.flow_weight_decay
        )

        self.load_flow(config.lfm_path)


    def train_step(self, states, actions):
        """
            states:  (B, T, 3, 128, 256)
            actions: (B, T, 3)
        """
        target = states[:, 1:]
        target = target.reshape(-1, *self.config.observation_shape)
        rssm_states = self.compute_rssm_states(states, actions)
        rssm_states = rssm_states.reshape(-1, self.config.recurrent_size + self.config.latent_size)

        """
            target:      (B*(T-1), 3, 128, 256)
            rssm_states: (B*(T-1), 768)
        """

        latent_flow_loss = self.compute_latent_flow_loss(target, rssm_states)

        self.optimizer.zero_grad()
        latent_flow_loss.backward()
        nn.utils.clip_grad_norm_(
            self.world_model_parameters,
            self.config.gradient_clip,
            self.config.gradient_norm_type
        )
        self.optimizer.step()


    @torch.no_grad()
    def compute_rssm_states(self, states, actions):
        """
            states:  (B, T, 3, 128, 256)
            actions: (B, T, 3)
        """
        B, T = states.shape[:2] 
        encoded_states = self.rssm.encoder(states)

        hidden = torch.zeros(B, self.config.recurrent_size, device=self.config.device)
        latent = torch.zeros(B, self.config.latent_size, device=self.config.device)
        hiddens = torch.zeros(B, T-1, self.config.recurrent_size, device=self.config.device)
        latents = torch.zeros(B, T-1, self.config.latent_size, device=self.config.device)

        for t in range(1, T):
            hidden = self.rssm.recurrent_model(hidden, latent, actions[:, t-1])

            if t == 1:
                latent, _ = self.rssm.representation_model(hidden, encoded_states[:, t])
            else:
                latent, _ = self.rssm.transition_model(hidden)

            hiddens[:, t-1] = hidden
            latents[:, t-1] = latent

        # hiddens shape: (B, T-1, 512)
        # latents shape: (B, T-1, 256)
        rssm_states = torch.cat([hiddens, latents], dim=-1) # (B, T-1, 768)
        return rssm_states


    def compute_latent_flow_loss(self, target, rssm_states):
        # pixel -> latent
        z_1 = self.vae.encoder(target) * self.scale
        B = z_1.shape[0]

        t = torch.randn(B, device=self.config.device)
        t = torch.sigmoid(t)
        t = t.clamp(0.001, 0.999)
        t_exp = t[:, None, None, None]

        z_0 = torch.randn_like(z_1)
        z_t = (1 - t_exp) * z_0 + t_exp * z_1
        v_target = z_1 - z_0

        v_predicted = self.flow_decoder(z_t, t, rssm_states)
        loss = F.mse_loss(v_predicted, v_target)
        return loss
    
    ######################## Utils #########################

    @torch.no_grad()
    def ode_sample(self, rssm_state, steps=25, method='heun', fixed_noise=None):
        B = rssm_state.shape[0]

        if fixed_noise is None:
            z = torch.randn(B, *self.config.latent_shape_vae, device=self.config.device)
        else:
            z = fixed_noise

        dt = 1.0 / steps
        decoder = self.ema.ema_model

        for i in range(steps):
            t_current = i / steps
            t = torch.full((B,), t_current, device=self.config.device)
            v1 = decoder(z, t, rssm_state)

            if method == 'heun' and i < steps - 1:
                z_next = z + v1 * dt
                t2 = torch.full((B,), t_current + dt, device=self.config.device)
                v2 = decoder(z_next, t2, rssm_state)
                z = z + (v1 + v2) * 0.5 * dt
            else:
                z = z + v1 * dt

        # latent → pixel
        x = self.vae.decode(z / self.scale)
        return x.clamp(0, 1)
    

    @torch.no_grad()
    def compute_latent_scale(self, frame_loader):
        self.vae.change_train_mode(train=False)
        zs = []
        for frames in frame_loader:
            frames = frames.to(self.config.device, non_blocking=True)
            zs.append(self.vae.encode(frames))
        zs = torch.cat(zs)
        self.scale = 1.0 / zs.std().item()
        print(f"  latent std = {1/self.scale:.4f} -> scale = {self.scale:.4f}")
        return self.scale
    

    def save_flow(self, epoch, save_dir):
        save_path = os.path.join(save_dir, f'latent_flow_ep{epoch}.pth')
        torch.save({
            'flow_decoder': self.flow_decoder.state_dict(),
            'ema'         : self.ema.ema_model.state_dict(),
            'ema_step'    : self.ema.step,
            'optimizer'   : self.optimizer.state_dict(),
            'scale'       : self.scale,
        }, save_path)
        print(f"Latent Flow Model saved: {save_path}")


    def load_flow(self, checkpoint_path):
        print(f"Loading Latent Flow: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.flow_decoder.load_state_dict(checkpoint['flow_decoder'])
        self.ema.ema_model.load_state_dict(checkpoint['ema'])
        self.ema.step = checkpoint['ema_step']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scale = checkpoint['scale']
        print("Latent Flow Checkpoint loaded successfully.")