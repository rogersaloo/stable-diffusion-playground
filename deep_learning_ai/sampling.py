import torch
import numpy as np
import matplotlib.pyplot as plt
# from ipython.display import HTML

from config import beta1, beta2, timesteps, device, save_dir, n_cfeat, n_feat, height
from unet import ContextUnet



# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
ab_t[0] = 1

# consruct model
nn_model = ContextUnet(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height)

def denoise_add_noise(x, t, pred_noise, z=None):
    """# helper function; removes the predicted noise 
    (but adds some noise back in to avoid collapse)

    Args:
        x (_type_): inputs
        t (_type_): timestep
        pred_noise (_type_): predicted noise from initial noise - pred
        z (_type_, optional): additional noise at each timestep.

    Returns:
        _type_: _description_
    """
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise *((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

#load model weights and set to eval mode
nn_model.load_state_dict(torch.load(f"{save_dir}/model_trained.pth"))
nn_model.eval()
print("Loaded in Model")

@torch.no_grad()
def sample_ddpm(n_sample, save_rate=20):
    samples = torch.randn(n_sample, 3, height, height).to(device)

    intermediate = []
    for i in range(timesteps,0, -1):
        print(f"sampling timestep {i:3d}", end="\r")

        #reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        #reset z to zero
        z = 0

        eps = nn_model(samples, t) #predict noise e_(x_t, t)
        samples = denoise_add_noise(samples, i, eps, z)
        if i%20==0 or i==timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())
    intermediate = np.stack(intermediate)
    return samples, intermediate


def visualize_generation():
    # visualize samples
    plt.clf()
    samples, intermediate_ddpm = sample_ddpm(32)
    animation_ddpm = plot_sample(intermediate_ddpm,32,4,save_dir, "ani_run", None, save=False)
    HTML(animation_ddpm.to_jshtml())