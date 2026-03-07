# Cookie-Run-AI-v2
Play in an AI-generated environment by learning the first stage of Cookie Run Ovenbreak, "The Witch's Oven."  

**Previous Version:**
https://github.com/Jeon-ChanYoung/Cookie-Run-AI

<br>

| Item | Detail |
|------|--------|
| **Observation Size** | 128×256 pixels |
| **Action Space** | 3 actions (None, Jump, Slide) |
| **Training Data** | 50 real gameplay videos (~48,000 frames) |

<br>
  
## Real
<img src="assets/real.gif" width="512"/>

<br>
  
## Fake (AI-generated)
<img src="assets/fake.gif" width="512"/>

#### Model Architecture & Improvements

This project is an **enhanced version of Cookie-Run-AI v1**, featuring significant architectural improvements over the original RSSM (Recurrent State-Space Model) implementation.  

| Feature | v1 | v2 |
|---------|----|----|
| **Training Approach** | End-to-end | Two-stage (VQ-VAE → RSSM) |
| **Latent Space** | Pixel-level reconstruction | Discrete latent tokens |
| **Encoder** | Standard CNN | Pre-trained VQ-VAE encoder |
| **Reconstruction Target** | Raw pixels | Latent representations |  


<br>

## Loss 
![Loss](assets/loss.png)  
```
Ep    1 | Recon: 99355.41 | MSE: 0.183514 | KL:  1.024
Ep    2 | Recon: 93150.23 | MSE: 0.057269 | KL:  0.601
Ep    3 | Recon: 91861.38 | MSE: 0.031047 | KL:  0.600
Ep    4 | Recon: 91526.65 | MSE: 0.024237 | KL:  0.951
Ep    5 | Recon: 91349.94 | MSE: 0.020642 | KL:  1.353
...
Ep  296 | Recon: 90469.11 | MSE: 0.002722 | KL:  5.413
Ep  297 | Recon: 90470.75 | MSE: 0.002755 | KL:  5.401
Ep  298 | Recon: 90468.18 | MSE: 0.002703 | KL:  5.264
Ep  299 | Recon: 90467.58 | MSE: 0.002691 | KL:  5.248
Ep  300 | Recon: 90469.53 | MSE: 0.002730 | KL:  5.282
```

<br>

## How to Run  
**1. Clone the repository and install dependencies:** 
```
git clone https://github.com/Jeon-ChanYoung/Cookie-Run-AI-v2.git
pip install -r requirements.txt
```

<br>

**2. Setup Pre-trained Model:**  
Download the pre-trained weights (vqvae_ep30.pth, rssm_ep150.pth) from the Releases page and place them in the directory structure as follows:  
```
model_params/
    └── rssm_ep100.pth
    └── vqvae_ep30.pth
```
If model_params does not exist, create it.  

<br>

**3. Run the main.py(FastAPI-based)**  
```
python main.py
```

<br>

## Simulation  
<img src="assets/simulation.gif" width="512"/>

- ⬆️ Arrow Up: Jump
- ⬇️ Arrow Down: Slide
- 🔄 R Key: Reset

Upon starting, the simulation randomly selects an initial image from the samples/oven_of_witch directory and begins an interactive imagination rollout.  

### Notes  
- The simulation is lightweight and runs smoothly on a **CPU**.
- You can adjust FPS by modifying **ACTION_REPEAT_INTERVAL** value in static/javascript.js.
- The training process follows this structure:
  
```python
# vqvae
for epoch in range(1, vqvae_train_epochs + 1):
    vqvae.change_train_mode(train=True)

    for batch_index, frames in enumerate(frame_loader, 1):
        frames = frames.to(device, non_blocking=True)
        loss, recon_l, vq_l, p_l, usage = vqvae.train_step(frames)

    print_losses()

    vqvae.step_scheduler()

    if epoch % 3 == 0:
        vqvae.save_vqvae(epoch, save_dir)
        # vqvae.visualize_recon(frame_loader)

# rssm
for epoch in range(1, rssm_train_epochs + 1):
    rssm.change_train_mode(train=True)

    for batch_index, (idx_batch, actions) in enumerate(rssm_loader, 1):
        idx_batch = idx_batch.to(device, non_blocking=True)
        actions   = actions.to(device, non_blocking=True)
        loss, recon_l, kl_l, acc  = rssm.train_step(idx_batch, actions)

    print_losses()

    if epoch % 5 == 0:
        rssm.save_rssm(epoch, save_dir)
        # rssm.visualize(vqvae, rssm_loader, epoch=epoch, n_frames=10, save_dir=vis_dir)
```
