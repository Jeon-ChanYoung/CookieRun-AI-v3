import torch
import numpy as np
import base64
import os
import cv2
import time

class Wrapper:
    def __init__(self, config, vqvae, rssm):
        self.config = config
        self.vqvae = vqvae
        self.rssm = rssm

        self.action_map = {
            "none": 0, 
            "jump": 1, 
            "slide": 2
        }

        self._load_samples()

        self._action_tensors = {
            name: self._create_action_tensor(idx)
            for name, idx in self.action_map.items()
        }
        self._zero_latent = torch.zeros(
            1, config.latent_size, device=config.device
        )
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        # recording
        self.enable_recording = config.enable_recording
        self.recording_dir = "recordings"
        if self.enable_recording:
            os.makedirs(self.recording_dir, exist_ok=True)
        self._video_writer = None
        self._frame_count = 0
        self._record_start = None

        self.reset()
        print("Game state initialized")


    def _start_recording(self, frame):
        if not self.enable_recording:
            return

        h, w = frame.shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.recording_dir, f"gameplay_{timestamp}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.config.video_fps
        self._video_writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        self._frame_count = 0
        self._record_start = time.monotonic()
        self._record_filename = filename
        print(f"🔴 Recording started: {filename}")

    def _save_recording(self):
        if not self.enable_recording or self._video_writer is None:
            return

        self._video_writer.release()
        self._video_writer = None

        elapsed = time.monotonic() - self._record_start
        print(
            f"⬜ Recording saved: {self._record_filename} "
            f"({self._frame_count} frames, {elapsed:.1f}s)"
        )
        self._frame_count = 0
        self._record_start = None


    def _record_frame(self, img_rgb):
        if not self.enable_recording or self._video_writer is None:
            return
        
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        self._video_writer.write(img_bgr)
        self._frame_count += 1


    def _create_action_tensor(self, action_idx):
        action = torch.zeros(1, self.config.action_size, device=self.config.device)
        action[0, action_idx] = 1.0
        return action


    @torch.inference_mode()
    def reset(self):
        self._save_recording()

        self.recurrent_state = torch.zeros(
            1, self.config.recurrent_size, device=self.config.device
        )

        initial_img = self.single_state_sample()         # (1, 3, 128, 256)
        initial_indices = self.vqvae.encode(initial_img) # (1, 8, 16) LongTensor

        encoded_state = self.rssm.encoder(initial_indices)    # (1, encoded_state_size)

        action = self._action_tensors["none"]

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            torch.zeros(1, self.config.latent_size, device=self.config.device),
            action,
        )

        self.latent_state, _ = self.rssm.representation_model(
            self.recurrent_state, encoded_state
        )
        img = self.get_current_image()

        self._start_recording(img)
        self._record_frame(img)

        return img

    @torch.inference_mode()
    def step(self, action_name: str):
        action_tensor = self._action_tensors.get(
            action_name, self._action_tensors["none"]
        )

        self.recurrent_state = self.rssm.recurrent_model(
            self.recurrent_state,
            self.latent_state,
            action_tensor,
        )
        self.latent_state, _ = self.rssm.transition_model(self.recurrent_state)
        
        img = self.get_current_image()
        self._record_frame(img)

        return img


    @torch.inference_mode()
    def get_current_image(self):
        token_logits = self.rssm.decoder(self.recurrent_state, self.latent_state)

        token_indices = token_logits.argmax(dim=-3)  

        reconstruction = self.vqvae.decode(token_indices)

        img = reconstruction[0].clamp_(0.0, 1.0).mul_(255.0).byte().cpu().numpy()
        img = np.ascontiguousarray(img.transpose(1, 2, 0)) # (C, H, W) -> (H, W, C)

        return img


    def image_to_base64(self, img):
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), self._jpeg_params)
        img_base64 = base64.b64encode(buf.data).decode("ascii")
        return f"data:image/jpeg;base64,{img_base64}"


    def single_state_sample(self):
        idx = np.random.randint(0, len(self.sample_images))
        img = self.sample_images[idx]  # (128, 256, 3) uint8

        img_tensor = torch.from_numpy(img).float() / 255.0
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

    def __del__(self):
        self._save_recording()
