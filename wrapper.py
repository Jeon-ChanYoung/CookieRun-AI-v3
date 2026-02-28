import os
import cv2
import base64
import torch
import numpy as np

from io import BytesIO
from PIL import Image
from modules.world_model import LFMWorldModel

class Wrapper:
    def __init__(self, config, world_model: LFMWorldModel):
        self.config = config
        self.world_model = world_model

        self.world_model.flow_decoder.eval()
        self.world_model.ema.ema_model.eval()
        
        self.action_map = {
            'none': 0,
            'jump': 1,
            'slide': 2,
        }

        self._load_samples()
        self._action_tensors = {
            name: self._create_action_tensor(idx)
            for name, idx in self.action_map.items()
        }

        self.reset()
        print("Game state initialized")

    def _create_action_tensor(self, action_idx):
        action = torch.zeros(1, self.config.action_size, device=self.config.device)
        action[0, action_idx] = 1.0
        return action
    
    @torch.no_grad()
    def reset(self):
        # recurrnt_size = 512
        self.recurrent_state = torch.zeros(1, self.config.recurrent_size, device=self.config.device)
    
        # observation_shape = (1, 3, 128, 256)
        initial_state = self.single_state_sample()

        encoded_state = self.world_model.rssm.encoder(initial_state)

        action_index = 0
        action_name = list(self.action_map.keys())[action_index]
        action = self._action_tensors[action_name]

        zero_latent = torch.zeros(
            1, self.config.latent_size, device=self.config.device
        )

        # latent_size = 16 * 16 = 256
        self.recurrent_state = self.world_model.rssm.recurrent_model(
            self.recurrent_state,
            zero_latent,
            action
        )

        self.latent_state, _ = self.world_model.rssm.representation_model(
            self.recurrent_state,
            encoded_state
        )

        return self.get_current_image()

    @torch.no_grad()
    def step(self, action_name: str):
        # action_tensor shape: (1, action_size)
        action_tensor = self._action_tensors.get(
            action_name, self._action_tensors['none']
        )

        self.recurrent_state = self.world_model.rssm.recurrent_model(
            self.recurrent_state,
            self.latent_state,
            action_tensor
        )

        self.latent_state, _ = self.world_model.rssm.transition_model(
            self.recurrent_state
        )

        return self.get_current_image()
    
    @torch.no_grad()
    def get_current_image(self):
        rssm_state = torch.cat(
            [self.recurrent_state, self.latent_state], dim=-1
        )

        reconstruction_img = self.world_model.ode_sample(
            rssm_state, 
            steps=25, 
            method='heun'
        )

        img = reconstruction_img[0].clamp(0, 1)
        img = (img * 255).byte().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        
        return img

    def image_to_base64(self, img):

        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        pil_img = Image.fromarray(img)
        buffered = BytesIO()
        pil_img.save(buffered, format="WEBP", quality=90)

        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{img_base64}"

    def single_state_sample(self):
        idx = np.random.randint(0, len(self.sample_images))
        img = self.sample_images[idx]  # (128, 256, 3) uint8
        
        img_tensor = torch.from_numpy(img).float() / 255.0  # [0, 1] 
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 256)
        img_tensor = img_tensor.to(self.config.device)
        
        return img_tensor

    def _load_samples(self):
        samples_dir = "samples/oven_of_witch"

        if not os.path.exists(samples_dir):
            raise FileNotFoundError(f"Samples directory not found: {samples_dir}")

        self.sample_images = []

        for filename in sorted(os.listdir(samples_dir)):
            if not filename.startswith("oow_sample") or not filename.endswith(".png"):
                continue

            file_path = os.path.join(samples_dir, filename)

            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.sample_images.append(img_rgb)

        print(f"✅ Loaded {len(self.sample_images)} sample images")