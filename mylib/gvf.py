from functools import partial
import torch
from torch.autograd.functional import jvp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Any, Optional, Union, List, Type 
from .helpers import Config

from training.networks import HookManager

class Register:
    _registry = {}

    @classmethod 
    def register(cls, name: str):
        def wrapper(config_cls):
            cls._registry[name] = config_cls
            return config_cls
        return wrapper

    @classmethod
    def registry(cls):
        """Get all registered config classes"""
        return cls._registry.copy()

@Register.register('base')
@dataclass(frozen=True)
class ConfigGVFBase(Config):
    template_path: Optional[str] = None
    flatten_input: bool = False
    threshold_weight: Optional[Union[float, List[float]]] = None
    threshold_time_min: Optional[Union[float, List[float]]] = None
    threshold_time_max: Optional[Union[float, List[float]]] = None

@Register.register('ambient')
@dataclass(frozen=True)
class ConfigGVFAmbient(ConfigGVFBase):
    scale: Union[float, List[float]] = 1.0
    decay_rate: Union[float, List[float]] = 1.0
    v_0: Optional[Union[float, List[float]]] = 30.0

@Register.register('linear')
@dataclass(frozen=True)
class ConfigGVFLinear(ConfigGVFAmbient):
    n_features: Optional[Union[int, List[int]]] = None
    dim_feature: Optional[Union[int, List[int]]] = None
    seed_mat: Optional[int] = None
    T: Optional[Union[int, List[int]]] = None

@Register.register('base_nonlinear')
@dataclass(frozen=True)
class ConfigGVFNonLinear(ConfigGVFBase):
    type_eval:  Optional[Literal["jvp", "numdiff"]] = None
    vf_latent:  Any = None
    step_size:  Optional[float] = None

@Register.register('hf')
@dataclass(frozen=True)
class ConfigGVFHuggingFace(ConfigGVFNonLinear):
    hf_url: Optional[str] = None

@Register.register('unet')
@dataclass(frozen=True)
class ConfigGVFUnet(ConfigGVFNonLinear):
    idx_skips: Optional[List[int]] = None

@Register.register('unet_attn')
@dataclass(frozen=True)
class ConfigGVFUnetAttention(ConfigGVFNonLinear):
    idxs: Optional[List[int]] = None

# --- Guidance Vector Fields ---

class AttentionMixture:
    def __init__(self, means, stds, weights_mixture):
        # means: (N, D), stds: (N,), weights_mixture: (N,)
        self.means = means
        self.stds = stds
        self.weights_mixture = weights_mixture
        self.D = means.size(-1)  # Dimension of the data

        # Validate that mixture weights sum to 1.0
        if not torch.isclose(torch.sum(weights_mixture), torch.tensor(1.0, dtype=weights_mixture.dtype), atol=1e-6):
            raise ValueError(f"weights_mixture must sum to 1.0, got sum={torch.sum(weights_mixture).item():.4f}")

    def __call__(self, x, T=1.0, passing_diff=False):
        """
        Args:
            x: Input tensor of shape (B, D) where B is batch size
            T: Temperature parameter (>0) controlling softmax sharpness
        Returns:
            weights_attention : Attention assocciated with gradient of log-density of mixture model, shape (B, D)
        """
        if passing_diff:
            diff = x 
        else:
            B, D = x.shape
            N, _ = self.means.shape

            # Compute squared distances between x and all means (B, N)
            diff = self.means.unsqueeze(0) - x.unsqueeze(1)  # (1, N, D) - (B, 1, D) → (B, N, D)


        mahalanobis = (diff ** 2).sum(dim=-1) / self.stds.unsqueeze(0)  # (B, N)
        energy_mahalana = -0.5 * mahalanobis  ** 2  # (B,N)

        # Compute log terms for each component 
        log_weights = torch.log(self.weights_mixture + 1e-8)                     # (N,)
        log_std_term = -self.D * torch.log(self.stds + 1e-8)                    # (N,)

        # Combine, drop the constant -(D/2)*ln(2π) 
        energy = energy_mahalana + log_weights.unsqueeze(0) + log_std_term.unsqueeze(0)  # (B,N)

        # Apply temperature and compute attention weights (B, N)
        weights_attn = F.softmax(T * energy, dim=-1)
        return weights_attn

class GuidanceVF:
    def flat(self, x):
        return rearrange(x, "... c h w -> ... (c h w)")
    
    def unflat(self, x):
        return rearrange(x, "... (c h w) -> ... c h w", c=self.templates.shape[-3], h=self.templates.shape[-2], w=self.templates.shape[-1])

    def __init__(self, templates, scale, v_0, decay_rate, latent, latent_inv,  *, 
                 flatten_input=False, 
                 threshold_weight=None,
                 threshold_time_min=None,
                 threshold_time_max=None,
                 attention=None,
                 **kwargs
                 ):

        # Core parameters
        self.templates = templates
        self.scale = scale
        self.v_0 = v_0
        self.decay_rate = decay_rate
        self.latent = latent
        self.latent_inv = latent_inv
        self.noise = lambda x: x 
        self.noise_dot = lambda x : 1 
        self.time_weight = lambda t: torch.sigmoid(self.decay_rate * (self.noise(t) - self.v_0)) 
        # Optional features
        self.flatten_input = flatten_input
        self.threshold_weight = threshold_weight
        self.threshold_time_min = threshold_time_min
        self.threshold_time_max = threshold_time_max

        self.attention = attention
        # Determine to use attention or not 
        if self.attention:
            self._score = self._score_attention
        else:
            self._score = self._score_single_feature

        # Pre-process templates 
        # TODO : ASSERT THIS IS (BATCH, N_FEATURES, D_LATENT)
        self.features_template = latent(self.flat(self.templates)) if flatten_input else latent(self.templates)
        # Device and type tracking
        self.device = self.templates.device
        self.dtype = self.templates.dtype
        # For testing
        self.history_weight = []
        self.history_apply_score = []
        self.history_attention = []
        
    def __call__(self, x, t):
        if self.flatten_input:
            x = self.flat(x) 
        
        if self.should_apply_score(t):
            dx_guidance = self.reverse_step(x, t)
        else:
            dx_guidance = torch.zeros_like(x)
        
        if self.flatten_input:
            dx_guidance = self.unflat(dx_guidance) 

        return dx_guidance

    def should_apply_score(self, t) -> bool:
        weight = torch.max(torch.sigmoid(self.decay_rate * (self.noise(t) - self.v_0)) * self.scale)
        apply_score = True
        # Check weight threshold
        if self.threshold_weight is not None and weight < self.threshold_weight:
            apply_score = False
        # Check time thresholds
        if self.threshold_time_min is not None and t < self.threshold_time_min:
            apply_score = False
        if self.threshold_time_max is not None and t > self.threshold_time_max:
            apply_score = False
        self.history_weight.append(weight)
        self.history_apply_score.append(apply_score)

        return apply_score 


    def reverse_step(self, x, t):
        return -self.noise_dot(t) * self.noise(t)  * self.score(x, t)

    def score(self, x, t):
        return self._score(x, t)

    def _score_single_feature(self, x, t):
        raise NotImplementedError("Subclasses must implement this method")

    def _score_attention(self, x, t):
        raise NotImplementedError("Subclasses must implement this method")

class AmbientGVF(GuidanceVF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, latent=lambda x: x, latent_inv=lambda x: x)

    def _score_single_feature(self, x, t):
        # (1, ) * (1, ) * ( (B, D) - (B,D))
        score = self.scale * self.time_weight(t) * (self.templates - x) / self.noise(t)**2 
        return score
    
    def _score_attention(self, x, t):
        # (B, N, D) - (B, 1, D)
        diffs = self.features_template - x.unsqueeze(1)
        diffs_normalized = self.attention_normalizer(diffs)
        # (B, N)
        attention = self.attention(diff_normalized, passing_diff=True)
        # (N, )
        weights =  attention * time_weight.unsqueeze(0) * self.scale
        score =  torch.einsum("BN, BN... -> B...", weights, recons) / self.noise(t) ** 2
        self.history_attention.append(attention)
        return score

class LinearGuidanceVF(GuidanceVF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # (N_u, N_f, L) -> (N_u * N_f, L) 
        self.features_template = self.features_template.flatten(0, 1)

    def _score_single_feature(self, x, t):
        # (1, ) * (1, ) * ( (B, D) - (B,D))
        features = self.latent(x)
        score = self.scale * self.time_weight(t) * self.latent_inv(self.features_templates - features) / self.noise(t)**2 
        return score

    def _score_attention(self, x, t):
        features = self.latent(x)
        # (B, N, D) - (B, 1, D)
        diffs = self.features_template - x.unsqueeze(1)
        diffs_normalized = self.attention_normalizer(diffs)
        # (B, N)
        attention = self.attention(diff_normalized, passing_diff=True)
        # (N, )
        weights =  attention * time_weight.unsqueeze(0) * self.scale
        score =  torch.einsum("BN, BN... -> B...", weights, recons) / self.noise(t) ** 2
        self.history_attention.append(attention)
        return score

    def _score_attention(self, x, t):
        # (B, F, L)
        features = self.latent(x)
        # (B, F * T, L)
        features = torch.repeat_interleave(features, dim=1, repeats=self.templates.shape[0])
        # (B, F * T, L) = (B, F * T, L) - (B, F * T, L) 
        diff_features = self.features_template - features
        # (B, F * T, D) 
        recons = self.latent_inv(diff_features)
        # TODO : make this a parameter
        diff_features_normalized = (diff_features - torch.mean(diff_features, dim=1, keepdim=True)) / torch.std(diff_features, dim=1,keepdim=True)
        # (B, F * T)
        attention = self.attention(diff_features_normalized, passing_diff=True)
        # (B, F * T) * (1, ) * (1, )
        weights = attention * self.time_weight(t) * self.scale 
        # (B, F * T)  , (B, F * T, D) -> (B, D)
        score =  torch.einsum("BN, BN... -> B...", weights, recons) / self.noise(t) ** 2
        self.history_weight.append(attention)
        return score

class NonLinearGuidanceVFBase():
    def __init__(self, vf_latent, latent, latent_inv):
        self.vf_latent = vf_latent
        self.latent = latent
        self.latent_inv = latent_inv
        self.noise = lambda x : x 
        self.noise_dot = lambda x : 1

    def __call__(self, x, t):
        if self.should_apply_score(t):
            dx_guidance = self.reverse_step(x, t)
        else:
            dx_guidance = torch.zeros_like(x)
        return dx_guidance

    def reverse_step(self, x, t):
        with torch.no_grad():
            x_latent = self.latent(x)
            score_latent = self.vf_latent.score(x_latent, t)
            score = self._pullback(x_latent, score_latent, t)
            # print(score_latent.sum(), score.sum())
            dx = -self.noise_dot(t) * self.noise(t) * score
        return dx

    def should_apply_score(self, t):
        return self.vf_latent.should_apply_score(t)
    
    def _pullback(self, x_latent, dx_latent, t):
        raise NotImplementedError("Subclasses must implement this method")

class JVPGuidanceVF(NonLinearGuidanceVFBase):
    def _pullback(self, x_latent, dx_latent, t):
        with torch.no_grad():
            _, dx = jvp(self.latent_inv, x_latent, dx_latent, strict=False)
        return dx

class NumericalGuidanceVF(NonLinearGuidanceVFBase):
    def __init__(self, *args, step_size=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size = step_size  

    def _pullback(self, x_latent, dx_latent, t):
        with torch.no_grad():
            step_size = self.step_size if self.step_size > 0.0 else t
            perturbed_latent = x_latent + step_size * dx_latent
            f_perturbed = self.latent_inv(perturbed_latent)
            f_original = self.latent_inv(x_latent)
            dx = (f_perturbed - f_original) / step_size  
        return dx 

# --- Builders --- 

@dataclass
class ContextBuilder:
    device      : Literal["cuda", "cpu"] = "cpu"
    dtype       : Literal[torch.float16, torch.float32, torch.float64] = torch.float64
    net         : Optional[Any] = None
    class_idx   : Optional[Any] = None

class BuilderBase(ABC):
    def __init__(self, config: ConfigGVFBase, templates: torch.Tensor, **context):
        self.config = config
        self.ctx = ContextBuilder(**context)
        self.latent_fn = None
        self.latent_inv_fn = None
        self.features_template = None
        self.device = context.get('device', "cpu")
        self.dtype = context.get("dtype", torch.float32)
        self.templates = templates
        self.batch_size = self.templates.shape[0]
        
    @abstractmethod
    @torch.no_grad
    def build(self) -> callable:
        pass

class BuilderAmbientGVF(BuilderBase):
    """Builder for ambient-space vector fields"""
    def build(self):
        return AmbientGVF(templates=self.templates, **self.config.to_dict() )
    
class BuilderNonLinearBase(BuilderBase):
    """Builder for pixel-space vector fields"""

    @abstractmethod
    def _setup_latents(self):
        """Method to setup the latent and latent inverse functions"""
        pass

    def _create_features_template(self):
        """Main method to create the latent features template"""
        return self.latent_fn(self.templates)

    def build(self):
        self._setup_latents()
        self.features_template = self._create_features_template()
        
        match self.config.type_eval:
            case "numdiff" : vf_class = NumericalGuidanceVF 
            case "jvp" : vf_class = JVPGuidanceVF
            case _: raise ValueError(f"Unknown eval type: {self.config.type_eval}")
        vf_latent = create_vf(self.config.vf_latent, self.features_template)
        
        vf = vf_class(
            vf_latent=vf_latent,
            latent=self.latent_fn,
            latent_inv=self.latent_inv_fn
        )

        if self.config.step_size is not None:
            vf.step_size = self.config.step_size

        return vf 

class BuilderHFGVF(BuilderNonLinearBase):
    def _setup_latents(self):
        from diffusers import AutoencoderKL
        
        vae = AutoencoderKL.from_pretrained(
            self.config.hf_url, 
            subfolder="vae",
            use_safetensors=True
        ).eval().requires_grad_(False).to(device=self.ctx.device, dtype=self.ctx.dtype)
        
        self.latent_fn = lambda x: vae.encode(x).latent_dist.sample()
        self.latent_inv_fn = lambda x: vae.decode(x).sample

class BuidlerUNetGVF(BuilderNonLinearBase):
    def _setup_latents(self, *, _test_sigma=None):
        # Init Hook Manager and attach to network
        hook_manager = HookManager()
        hook_manager.save_fwd = True
        hook_manager.save_skips = True
        self.ctx.net.hook_manager = hook_manager
        # Run network once to generate skips
        if _test_sigma is None:
            _test_sigma = torch.tensor(10).to(device=self.device, dtype=self.dtype)
        self.ctx.net(self.templates, _test_sigma)
        #
        idx_mod = list(self.config.idx_skips)
        total_skips = len(hook_manager.saved_skips)
        idx_preserved = [i for i in range(0, total_skips) if i not in idx_mod] 
        shapes_skips = [hook_manager.saved_skips[i].shape[1:] for i in idx_mod]

        def latent_fn(x, *, net):
            # Only capture modified skips
            skips = [net.hook_manager.saved_skips[i] for i in idx_mod]
            skips_flat = torch.cat([x.flatten(start_dim=1) for x in skips], dim=1)
            return skips_flat

        def latent_inv_fn(z, *, net):
            # Split latent vector
            lengths = [torch.prod(torch.tensor(shape)) for shape in shapes_skips]
            splits = z.split(lengths, dim=1)
            skips_modded = [
                split.view(-1, *shape) 
                for split, shape in zip(splits, shapes_skips)
            ]

            # Recombine skips 
            full_skips = [] 
            for i in range(total_skips):
                if i in idx_mod:
                    full_skips.append(skips_modded[idx_mod.index(i)])
                else: 
                    full_skips.append(net.hook_manager.saved_skips[i])
            
            # Middle information
            emb = net.hook_manager.saved_emb
            z = full_skips[-1]

            # Decoder.
            for block in self.ctx.net.model.dec.values():
                if z.shape[1] != block.in_channels:
                    z = torch.cat([z, full_skips.pop()], dim=1)
                z = block(z, emb)
            F_x = net.model.out_conv(torch.nn.functional.silu(net.model.out_norm(z)))

            x, sigma = net.hook_manager.fwd_vars.x, net.hook_manager.fwd_vars.sigma

            c_skip = net.sigma_data ** 2 / (sigma ** 2 + net.sigma_data ** 2)
            c_out = sigma * net.sigma_data / (sigma ** 2 + net.sigma_data ** 2).sqrt()

            D_x = c_skip * x + c_out * F_x
            return D_x

        self.latent_fn = partial(latent_fn, net=self.ctx.net)
        self.latent_inv_fn = partial(latent_inv_fn, net=self.ctx.net)

    def _create_features_template(self):
        # Determine class labelsj
        if self.ctx.class_idx is None:
            # Use zeros (no class)
            class_labels = torch.zeros([self.batch_size, self.ctx.net.label_dim], device=self.device)
        elif isinstance(self.ctx.class_idx, int):
            # Use one-hot encoded specified class
            class_labels = torch.eye(self.ctx.net.label_dim, device=self.device)[self.batch_size * [class_idx]]
        elif torch.is_tensor(self.ctx.class_idx):
            # Use provided class labels directly
            assert self.ctx.class_idx.shape == (self.batch_size, self.ctx.net.label_dim), \
                f"class_labels must have shape [{batch_size}, {net.label_dim}]"
            class_labels = self.ctx.class_idx.to(self.device)
        else:
            raise ValueError("class_idx must be None, int, or tensor")

        sigma = torch.tensor(1e-1).to(self.device).to(self.dtype)
        self.ctx.net(self.templates, sigma, class_labels)
        features_template = self.latent_fn(None)
        assert not torch.any(torch.isnan(features_template)), "features_template has NaNs"
        return features_template

class BuilderUNetAttentionGVF(BuilderNonLinearBase):
    def _setup_latents(self):
        # Initialize Hook Manager
        hook_manager = HookManager()
        hook_manager._is_parallel = True
        # Register which blocks get modified
        name_blocks_with_attention = []
        for name, block in self.ctx.net.model.enc.items():
            # Check if block uses attention
            if getattr(block, "num_heads", 0) > 0:
                # Log the name 
                name_blocks_with_attention.append(name)
        assert len(name_blocks_with_attention) == 9

        for i in self.config.idxs:
            hook_manager.register(name_blocks_with_attention[i])
        assert len(hook_manager._registered_names) == len(self.config.idxs)

        self.ctx.net.hook_manager = hook_manager
        hook_manager.save_blocks = True
        hook_manager.save_fwd = True

        def latent_fn(x, *, net):
            z = [hook_manager.load(name) for name in hook_manager._registered_names]
            z = torch.stack(list(z), dim=1) # TODO: NO CLUE IF THE DIMENSIONALITY OF THIS MAKES SENSE FOR OUR CODE
            return z

        def latent_inv_fn(z, *, net):
            """Make sure to check whether you're running sequential or parallel"""
            # Load in guided attention
            for name, attn in zip(hook_manager.registered_names, torch.unbind(z, dim=1)):
                hook_manager.save(name, attn)
            # Run model with hybrid attention
            x, sigma, class_labels, force_fp32, model_kwargs = hook_manager.dump_fwd()
            y = net(x, sigma, class_labels, force_fp32, **model_kwargs)
            # Re-enable saving, Disable loading
            return y

        self.latent_fn = partial(latent_fn, net=self.ctx.net)
        self.latent_inv_fn = partial(latent_inv_fn, net=self.ctx.net)

    def _create_features_template(self):
        # Determine class labelsj
        if self.ctx.class_idx is None:
            # Use zeros (no class)
            class_labels = torch.zeros([self.batch_size, self.ctx.net.label_dim], device=self.device)
        elif isinstance(self.ctx.class_idx, int):
            # Use one-hot encoded specified class
            class_labels = torch.eye(self.ctx.net.label_dim, device=self.device)[self.batch_size * [class_idx]]
        elif torch.is_tensor(self.ctx.class_idx):
            # Use provided class labels directly
            assert self.ctx.class_idx.shape == (self.batch_size, self.ctx.net.label_dim), \
                f"class_labels must have shape [{batch_size}, {net.label_dim}]"
            class_labels = self.ctx.class_idx.to(self.device)
        else:
            raise ValueError("class_idx must be None, int, or tensor")

        sigma = torch.tensor(1e-1).to(self.device).to(self.dtype)
        self.ctx.net(self.templates, sigma, class_labels)
        features_template = self.latent_fn(None).detach().clone()
        assert not torch.any(torch.isnan(features_template)), "features_template has NaNs"
        return features_template

def create_vf(cfg: ConfigGVFBase, templates: torch.Tensor, verbose: bool = True, **ctx) -> callable:
    if verbose:
        print(f"\nConfig: {cfg}")
        print(f"Templates shape: {templates.shape}")
    
    if cfg is None:
        return lambda x, t: torch.zeros_like(x)

    if type(cfg) is ConfigGVFAmbient:           builder = BuilderAmbientGVF(cfg, templates, **ctx)
    elif type(cfg) is ConfigGVFLinear:          builder = BuilderLinearVF(cfg, templates, **ctx)
    elif type(cfg) is ConfigGVFHuggingFace:     builder = BuilderHFGVF(cfg, templates, **ctx)
    elif type(cfg) is ConfigGVFUnet:            builder = BuidlerUNetGVF(cfg, templates, **ctx)
    elif type(cfg) is ConfigGVFUnetAttention:   builder = BuilderUNetAttentionGVF(cfg, templates, **ctx)
    else: raise ValueError(f"Unknown config type: {type(cfg).__name__}")

    return builder.build()

def args_is_gvf(args):
    return args.keys().issubset(["latent", "vectorfield", "latent_inv", "evaluator"])

def match_args_to_latent(args):
    match args:
        case "ambient":
            return "ambient"
        case {"dim_in": _, "dim_out": _, "n_features" :_}:
            return "linear"

def match_args_to_evaluator(args):
    match args:
        case "jvp":
            return "ambient"
        case {"dim_in": _, "dim_out": _, "n_features" :_}:
            return "linear"

def match_args_to_vectorfield(args):

def args_is_nonlinear(args):
    nonlinear_types = ["unet", "hf"]
    latent_type = match_args_to_latent(args)
    return latent_type in nonlinear_types

def create_gvf_from_dict(latent: dict, vectorfield: dict, 
                         latent_inv: Optional[dict]=None , evaluator: Optional[dict]=None):

    # Create vectorfield 
    if args_is_gvf(latent):
        vf_latent = create_gvf_from_dict(vectorfields)
    else:
        vf_latent = GuidanceVF(vectorfields)

    # Get type of latent
    type_latent = match_args_to_latent(latent)

    # Determine if evaluation operator is necessary
    if is_nonlinear(type_latent):
        evaluator = 
        gvf = GuidanceVF(...)
    elif args_evaluator is not None:
        raise ValueError(f"Evaluator args can only be passed when type_latent is nonlinear, \n" 
                         f"got type latent : {type_latent} and args_evaluator : args_evaluator} ")
    else:
        gvf = GuidanceVF(...)

    return gvf


