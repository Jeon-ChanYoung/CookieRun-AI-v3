# Cookie-Run-AI
Play in an AI-generated environment by learning the first stage of Cookie Run Ovenbreak, "The Witch's Oven."

- Observation Size: 128×256 pixels  
- Action Space: 3 actions (None, Jump, Slide)  
- Training Data: 31 real gameplay videos (approximately 27,000 frames)

<br>
  
## Real
<img src="assets/real.gif" width="512"/>  

<br>
  
## Fake (AI-generated)
<img src="assets/fake.gif" width="512"/>  

#### Model Architecture & Quality Note

This project utilizes a **Recurrent State-Space Model (RSSM)**. Please note that the generation quality is currently limited due to the relatively low resolution and small size of the training dataset. The training data consists of state-action sequences. Transitions involving "Game Over" screens or progression to subsequent stages were **excluded** from the training set to maintain continuity within the specific stage.  

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
git clone [your-repo-url]
pip install -r requirements.txt
```

<br>

**2. Setup Pre-trained Model:**  
Download the pre-trained weights (oow_ep300.pth) from the Releases page and place them in the directory structure as follows:  
```
model_params/
    └── oow_ep300.pth
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
