import torch
from config import beta1, beta2, timesteps, device, n_cfeat, n_feat, height
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

