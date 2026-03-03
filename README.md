# Cookie-Run-AI-v2
Play in an AI-generated environment by learning the first stage of Cookie Run Ovenbreak, "The Witch's Oven."  

**Previous Version:**
https://github.com/Jeon-ChanYoung/Cookie-Run-AI

<br>

| Item | Detail |
|------|--------|
| **Observation Size** | 128×256 pixels |
| **Action Space** | 3 actions (None, Jump, Slide) |
| **Training Data** | 31 real gameplay videos (~27,000 frames) |

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
MSE represented the reconstruction loss of the “Gaussian log-likelihood” as the loss of MSE. 

<br>

## How to Run  
**1. Clone the repository and install dependencies:** 
```
git clone https://github.com/Jeon-ChanYoung/Cookie-Run-AI.git
pip install -r requirements.txt
```

<br>

**2. Setup Pre-trained Model:**  
Download the pre-trained weights (oow_ep300.pth) from the Releases page and place them in the directory structure as follows:  
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
for episode in range(start_episodes, episodes + 1):

    for _ in range(world_model_update_step): # # world_model_update_step = 50
        # experiences = memory.sample_mixed(batch_size, batch_length, global_step, balanced_ratio=5)
        experiences = memory.sample(batch_size, batch_length)
        recon_loss, kl_loss = model.dynamic_learning(experiences)

    print_losses()

    if episode % 10 == 0:
        save_model_params(episode, save_dir)
```

<br> 

## References  
- https://github.com/danijar/dreamerv3
