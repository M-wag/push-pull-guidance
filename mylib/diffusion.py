import tqdm
import numpy as np
import torch 
import dnnlib
from mylib.gvf import BuilderPushPullVF

DISABLE_TQDM = False

#----------------------------------------------------------------------------
# Modified version torch.randn_like
def randn_like(tensor, generator=None):
    return torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=tensor.dtype,
        generator=generator
    )

#----------------------------------------------------------------------------
# Modified version of EDM sampler.
# Modular options for gradient and time step discretization.

class EDMSampler:
    @dnnlib.util.capture_init
    def __init__(
        self,
        device              ,
        num_steps           : int, 
        sigma_min           : float,
        sigma_max           : float, 
        rho                 : float, 
        S_churn             : float, 
        S_min               : float, 
        S_max               : float,
        S_noise             : float, 
        seed_dynamics       : int = None,
        dtype               = torch.float32,
        apply_2nd_order     : bool = True,
        save_all_timesteps  : bool = True,
        correct_rgb         : bool = False,
        disable_tqdm        : bool = False,
    ):

        self.num_steps = num_steps          
        self.sigma_min = sigma_min         
        self.sigma_max = sigma_max        
        self.rho = rho             
        self.S_churn = S_churn        
        self.S_min = S_min         
        self.S_max = S_max        
        self.S_noise = S_noise     

        self.seed_dynamics = seed_dynamics 
        self.device = device              
        self.dtype = dtype               
        self.apply_2nd_order = apply_2nd_order    
        self.save_all_timesteps = save_all_timesteps
        self.correct_rgb = correct_rgb      
        self.disable_tqdm = disable_tqdm    
        self.time_step_fn = time_steps_edm

    def init_gradient(self, gradient_kwargs, *, net=None):
        score_fn = ScoreDenoise(denoiser=net, scale=gradient_kwargs.scale_model_score)
        if "gvf" in gradient_kwargs.keys():
            score_fn = ScoreAdditive(score_fn, gradient_kwargs.gvf)
# Enable status prints?
        self.gradient_fn = GradientEDM(score_fn)

    @torch.no_grad()
    def __call__(self, dynamics, noise, labels):

        # load in object properties for lceared code
        num_steps, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise = (
            self.num_steps, self.sigma_min, self.sigma_max, self.rho, self.S_churn, self.S_min, self.S_max, self.S_noise 
        )

        # Adjust noise levels based on what's supported by the network
        sigma_min = max(sigma_min, dynamics.sigma_min)
        sigma_max = min(sigma_max, dynamics.sigma_max)
        
        # Time step discretization.
        t_steps = self.time_step_fn(num_steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho) # t_N=0
        t_steps = t_steps.to(device=self.device, dtype=self.dtype)

        xs = None 
        # Intialize empty array to save intermediate timesteps
        if self.save_all_timesteps:
            xs = torch.empty((num_steps, noise.shape[0], dynamics.img_channels, dynamics.img_resolution, dynamics.img_resolution))

        rng_generator_dynamics = self.seed_dynamics
        if self.seed_dynamics is not None:
            rng_generator_dynamics = torch.Generator(device=self.device).manual_seed(self.seed_dynamics)

        # Main sampling loop.
        x_next = noise.to(self.dtype) * t_steps[0]
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step', position=1, disable=self.disable_tqdm): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if (S_min <= t_cur <= S_max) else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur, rng_generator_dynamics)

            # Euler step
            d_cur = dynamics(x_hat, t_hat, labels)
            x_next = x_hat + d_cur * (t_next - t_hat)

            # Apply 2nd order correction
            if self.apply_2nd_order and i < num_steps - 1:
                d_prime = dynamics(x_next, t_next, labels)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            
            if self.save_all_timesteps:
                xs[i] = x_next

        x_0 = xs if self.save_all_timesteps else x_next
        if self.correct_rgb:
            x_0 = (x_0 * 127.5 + 128) / 255

        return x_0, (t_steps.cpu().numpy(), )
    
#----------------------------------------------------------------------------
# Time step discretization functions 

def time_steps_edm(num_steps, *, sigma_min, sigma_max, rho):
        if num_steps == 1:
            t_steps = torch.tensor([sigma_max, 0 ])
        else:
            step_indices = torch.arange(num_steps, dtype=torch.float64)
            t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        return t_steps

def time_steps_ddim(num_steps, device, net=None):
    raise NotImplementedError

#----------------------------------------------------------------------------
# Different gradients useable by the EDM sampler
# dx/dt = gradient(x, t)

class GradientEDM(torch.nn.Module):
    def __init__(self, score_fn, encoder):
        super().__init__()
        self.noise = lambda t : t
        self.noise_dot = lambda t : 1
        self.score_fn = score_fn
        self.encoder = encoder

    def forward(self, x, t, labels):
        return -self.noise(t) * self.noise_dot(t) * self.score_fn(x, t, labels)

    def update(self, state):
        examples_encoded = self.encoder.encode_latents(state.examples)
        self.score_fn.update(examples_encoded)

    def __getattr__(self, name):
        # try PyTorch's normal attribute resolution 
        try:
            return super().__getattr__(name)  # preserves all nn.Module magic
        except AttributeError:
            pass  # Not found → continue with your own logic
        # Check if network has propert
        if hasattr(self.score_fn, 'denoiser') and hasattr(self.score_fn.denoiser, name):
            return getattr(self.score_fn.denoiser, name)
        # Raise usual error
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

class ScoreDenoise(torch.nn.Module):
    def __init__(self, denoiser, *, scale=1.0, score_add=None, updater=None):
        super().__init__()
        self.denoiser = denoiser
        self.scale = scale
        self.score_add = score_add
        self._updater = updater

    def forward(self, x, noise, labels):
        denoised = self.denoiser(x, noise, labels).to(noise.dtype)
        score = self.scale * (denoised - x) / noise ** 2
        if self.score_add:
            score += self.score_add(x, noise)
        return score

    def update(self, examples):
        if callable(self._updater):
            self.score_add = self._updater(examples)

def create_dynamics(net, encoder, *, scale=1.0, ppvf_kwargs=None, device="cuda"):
    updater = None
    if ppvf_kwargs:
        builder_ppvf = BuilderPushPullVF(ppvf_kwargs)

        def _updater(examples):
            builder_ppvf.set_examples(examples.unsqueeze(1).to(device=device, dtype=torch.float32)) # should change automatically and shape should match N examples 
            return builder_ppvf.build(device=device)

        updater = _updater

    score_fn = ScoreDenoise(net, scale=scale, updater=updater)
    return GradientEDM(score_fn, encoder)
    
