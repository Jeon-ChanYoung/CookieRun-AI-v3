# Cookie-Run-AI-v3
Play in an environment where an AI learns the first stage of Cookie Run, “The Witch's Oven,” and generates the next screen in real time based on your input.  

**Previous Version:**  
- https://github.com/Jeon-ChanYoung/Cookie-Run-AI
- https://github.com/Jeon-ChanYoung/Cookie-Run-AI-v2
- 
<br>

| Item | Detail |
|------|--------|
| **Observation Size** | 128×256 pixels |
| **Action Space** | 3 actions (None, Jump, Slide) |
| **Training Data** | 50 real gameplay videos (~48,000 frames) |

<br>

## Training Data Distribution

> **Total Frames: 47,704** (from 50 real gameplay videos)

| Action | Label | Frames | Ratio |
|:------:|-------|-------:|------:|
| 0 | None | 35,773 | 75.0% |
| 1 | Jump | 1,249 | 2.6% |
| 2 | Slide | 10,682 | 22.4% |

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
| **Training Approach** | End-to-end | Two-stage (VQ-VAE -> RSSM) |
| **Latent Space** | Pixel-level reconstruction | Discrete latent tokens |
| **Encoder** | Standard CNN | Pre-trained VQ-VAE encoder |
| **Reconstruction Target** | Raw pixels | Latent representations |  

v2 transitions to a two-stage architecture consisting of a VQ-VAE and an RSSM.  

In the first stage, the VQ-VAE incorporates a pre-trained VGG16 network to compute perceptual loss, compressing raw 128x256 images into a discrete 16x32 grid of integer tokens based on a codebook of size 256. 

In the second stage, the RSSM models world dynamics by treating the next-step prediction as a categorical classification task over these discrete tokens rather than a continuous regression task. 

To optimize the training pipeline, the entire image dataset is pre-computed and converted into lightweight integer tokens using the trained VQ-VAE prior to the RSSM phase.  

<br>

## Loss 

### VQ-VAE Loss
<img src="assets/vqvae_recon.png" alt="recon" width="600">

```
Epoch [ 1/30] VQ-VAE loss: 0.203241  recon: 0.014470  vq_l: 0.010847  p_l: 1.779250  usage: 1.0
Epoch [ 2/30] VQ-VAE loss: 0.182763  recon: 0.012679  vq_l: 0.011716  p_l: 1.583680  usage: 1.0
Epoch [ 3/30] VQ-VAE loss: 0.167650  recon: 0.011198  vq_l: 0.011293  p_l: 1.451597  usage: 1.0
Epoch [ 4/30] VQ-VAE loss: 0.159327  recon: 0.010206  vq_l: 0.011196  p_l: 1.379252  usage: 1.0
Epoch [ 5/30] VQ-VAE loss: 0.159627  recon: 0.010195  vq_l: 0.012017  p_l: 1.374150  usage: 1.0
...
Epoch [26/30] VQ-VAE loss: 0.136206  recon: 0.007938  vq_l: 0.011916  p_l: 1.163512  usage: 1.0
Epoch [27/30] VQ-VAE loss: 0.136079  recon: 0.007839  vq_l: 0.012115  p_l: 1.161231  usage: 1.0
Epoch [28/30] VQ-VAE loss: 0.136079  recon: 0.007839  vq_l: 0.012115  p_l: 1.161231  usage: 1.0
Epoch [29/30] VQ-VAE loss: 0.135549  recon: 0.007781  vq_l: 0.011987  p_l: 1.155040  usage: 1.0
Epoch [30/30] VQ-VAE loss: 0.134972  recon: 0.007722  vq_l: 0.011902  p_l: 1.149331  usage: 1.0
```

<br>

### RSSM Loss
<img src="assets/rssm_recon.png" alt="recon" width="600">

<br>

<img src="assets/rssm_kl.png" alt="kl" width="600">

<br>

<img src="assets/rssm_acc.png" alt="acc" width="600">

```
Epoch [  1/100] RSSM loss: 3.428530  recon: 3.415260  kl: 1.327025  acc: 0.3446
Epoch [  2/100] RSSM loss: 2.967180  recon: 2.941917  kl: 2.526307  acc: 0.4186
Epoch [  3/100] RSSM loss: 2.726436  recon: 2.695507  kl: 3.092921  acc: 0.4796
Epoch [  4/100] RSSM loss: 2.506273  recon: 2.477390  kl: 2.888293  acc: 0.5286
Epoch [  5/100] RSSM loss: 2.510147  recon: 2.480252  kl: 2.989504  acc: 0.5280
...
Epoch [ 96/100] RSSM loss: 1.803927  recon: 1.768754  kl: 3.517291  acc: 0.7188
Epoch [ 97/100] RSSM loss: 1.845641  recon: 1.801281  kl: 3.436019  acc: 0.7082
Epoch [ 98/100] RSSM loss: 1.817078  recon: 1.783470  kl: 3.360827  acc: 0.7163
Epoch [ 99/100] RSSM loss: 1.858833  recon: 1.823287  kl: 3.554642  acc: 0.7027
Epoch [100/100] RSSM loss: 1.825670  recon: 1.791427  kl: 3.424293  acc: 0.7124
```

Here, "Accuracy" refers to the ratio of predicted VQ token indices that match the actual VQ tokens.  

<br>

## How to Run  
**1. Clone the repository and install dependencies:** 
```
git clone https://github.com/Jeon-ChanYoung/Cookie-Run-AI-v2.git
pip install -r requirements.txt
```

<br>

**2. Setup Pre-trained Model:**  
Download the pre-trained weights (vqvae_ep30.pth, rssm_ep100.pth) from the Releases page and place them in the directory structure as follows:  
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
<!-- <img src="assets/simulation.gif" width="600"/> -->

- ⬆️ Arrow Up: Jump
- ⬇️ Arrow Down: Slide
- 🔄 R Key: Reset
 
    if epoch % 5 == 0:
        rssm.save_rssm(epoch, save_dir)
        # rssm.visualize(vqvae, rssm_loader, epoch=epoch, n_frames=10, save_dir=vis_dir)
```
