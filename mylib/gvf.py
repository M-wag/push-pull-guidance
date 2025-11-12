import torch
import dnnlib

from typing import Optional, Tuple
from einops import rearrange
from functools import partial
from training.networks import InjectionManager

#----------------------------------------------------------------------------
# Some values in config will be turned to strings. The following map defines
# the deserialization for objects commonly found in configurations.

DESERIALIZATION_MAP = {
        "torch.float16" : torch.float16,
        "torch.float32" : torch.float32,
        "torch.float64" : torch.float64,
        "tensor([], size=(1, 0, 0, 0))" : torch.zeros(1, 0, 0, 0),
    }

#----------------------------------------------------------------------------
# Temporary Logger globa

LOGGER = {}

#----------------------------------------------------------------------------
# Helper function to compute the entropy from a probablity mass of shape (batch, p(x_i))

def entropy_from_mass(x):
    x = x.clamp_min(1e-12).flatten(start_dim=1)
    return -(x * torch.log2(x)).sum(dim=1)
#----------------------------------------------------------------------------
# Helper function to assign buffers for arguments that 'may' be torch.Tensor

def _maybe_register_buffer(module, name, value):
    if value is not None:
        module.register_buffer(name, torch.tensor(value))
    else:
        setattr(module, name, None)

#----------------------------------------------------------------------------
# Helper function to assign get pretty-printed attributes.

def _get_obj_attribute_as_lines(obj, names):
    lines = []
    for name in names:
        lines.append(f"  ({name}): {getattr(obj, name)}")
    return "\n".join(lines)

#----------------------------------------------------------------------------
# Sigmoidal time gating function which can be either quadratic or logistic

class NoiseGate(torch.nn.Module):
    def __init__(self, type_gate: str, nu: float, decay_rate: float = None,  noise_onset : float = float('inf'), decimals: int =4):
        super().__init__()
        self.type_gate = type_gate  
        self.decimals=decimals
        self.register_buffer("nu", torch.tensor(nu))
        self.register_buffer("noise_onset", torch.tensor(noise_onset))

        if type_gate == "logistic":
            if decay_rate is None: 
                raise ValueError("decay_rate must be provided for logistic gating")
            self._gate = self._logistic_gate
        elif type_gate == "quadratic":
            self._gate = self._quadratic_gate
        elif type_gate == "heaviside":
            self._gate = self._heaviside_gate
        else:
            raise ValueError(f"Unknown gating type: {type_gate!r}")

    def forward(self, noise):
        noise = self.round(noise)
        if noise > self.round(self.noise_onset):
            return torch.zeros_like(noise)
        return self._gate(noise)

    def _logistic_gate(self, noise):
          return torch.sigmoid(self.decay_rate * (noise - self.nu))

    def _quadratic_gate(self, noise):
        return noise**2 / (noise**2 + self.nu**2)

    def _heaviside_gate(self, noise):
        if noise >= self.round(self.nu):
            return torch.ones_like(noise)
        else:
            return torch.zeros_like(noise)
    
    def round(self, x):
        return torch.round(x.to(torch.float32), decimals=self.decimals)
    
    def extra_repr(self):
        return _get_obj_attribute_as_lines(self, ["type_gate", "decimals", "nu", "noise_onset"])


#----------------------------------------------------------------------------
# Pullback Operation evaluated by Numerical Differentiation

class PullbackNumericalDifferentiation(torch.nn.Module):
    def __init__(self, step_size_slope, step_size_intercept):
        super().__init__()
        self.register_buffer("slope", torch.tensor(step_size_slope))
        self.register_buffer("intercept", torch.tensor(step_size_intercept))
    
    def step_size(self, noise):
        return  noise**2 * self.slope + self.intercept

    @torch.no_grad
    def forward(self, latent_inv, x_latent, dx_latent, noise):
        perturbed_latent = x_latent + self.step_size(noise) * dx_latent
        f_perturbed = latent_inv(perturbed_latent, noise)
        f_original = latent_inv(x_latent, noise)
        dx = (f_perturbed - f_original) / self.step_size(noise)
        return dx

#----------------------------------------------------------------------------
# Attention weights for the score of mixture of Gaussians
# Essentialy performs softmax(-1/2 Mahalanobis(x)^2 + ln(weight) + ln(Z))
# However in our case Z and weighting are equal so can be ignored.
# Includes optional modificaitons like Temperature parameter and normalization before softmax
# See equation ..

class AttentionMixture(torch.nn.Module):
    def __init__(self, means, std, T=1.0, eps=1e-8, flat_data=False, 
                 normalize=False, pass_diff=False):
        super().__init__()
        self.register_buffer("means", means)    # (B, N, D)
        self.std = std
        self.T = T
        self.eps = eps

        self.flat_data = flat_data
        self.normalize = normalize
        self.pass_diff = pass_diff

        self.N, self.D = self.means.shape[0], self.means.shape[1]


    def forward(self, x):
        # Option to directly pass the difference between means and x
        diff = x 
        if not self.pass_diff:
            diff = self.means - x.unsqueeze(1)  # (B, N, D) 
        assert diff.shape == self.means.shape, f"Expect diff shape : {self.means.shape}, got : {diff.shape}"

        # Flatten data
        if self.flat_data:
            diff = torch.flatten(diff, start_dim=2)

        # Difference between x and means has variance = 1 
        if self.normalize:
            dim = (1,2)
            diff = diff - diff.mean(dim=dim, keepdim=True)
            diff = diff / (diff.std(dim=dim, keepdim=True) + self.eps)

        # squared Mahalanobis distance
        mahalanobis_sq = (diff.pow(2).sum(dim=-1)) / (self.std ** 2 + self.eps) # (B, N) 
        # Compute attention weight
        logits = -1/2 * 1/self.T * mahalanobis_sq # (B, N) 
        attn = torch.nn.functional.softmax(logits, dim=-1) # (B, N) 

        assert attn.shape == self.means.shape[:2] , f"Expect attention shape : {self.means.shape[:2]}, got : {tuple(attn.shape)}"
        return attn

#----------------------------------------------------------------------------
# Score for a noise-gated and diffused mixture of Dirac-Delta functions 

class ScoreGatedDiracMixture(torch.nn.Module):
    def __init__(
            self, 
            means, 
            noise_gate, 
            noise, 
            *,
            channeled = False,
            attention_kwargs = {},
        ):
        super().__init__()

        self.register_buffer("means", means)
        self.noise = noise 
        self.noise_gate = noise_gate
        self.channeled = channeled

        self.n_modes = self.means.shape[1] 

        if self.channeled:
            self._score = self._score_for_mixture_channeled
            self.attention = AttentionMixture(
                    rearrange(self.means, "B N C ... -> B (N C) ... "), 
                    self.noise_gate.nu, 
                    flat_data=True,
                    pass_diff=True,
                    **attention_kwargs)
        elif self.n_modes == 1:
            self._score = self._score_single_component
            self.attention = None
        else:
            self._score = self._score_for_mixture
            self.attention = AttentionMixture(
                    self.means, 
                    self.noise_gate.nu, 
                    flat_data=True,
                    pass_diff=True,
                    **attention_kwargs)

    def forward(self, x, t):
        if self.should_eval(x, t):
            return self._score(x, t)
        else:
            return torch.zeros_like(x)

    def should_eval(self, x, t):
        if self.noise_gate(t) == 0:
            return False
        return True

    def _score_single_component(self, x, t):
        means_flat = self.means.squeeze(1)  # (B, 1, *D) -> (B, *D)
        noise = self.noise(t)
        score = self.noise_gate(noise) * (means_flat - x) / noise**2
        return score

    def _score_for_mixture(self, x, t):
        dif_x_to_mu = self.means - x.unsqueeze(1)                   # (B, N, *D)
        attn = self.attention(dif_x_to_mu)       # (B, N)
        noise = self.noise(t)
        weights = attn * self.noise_gate(noise).unsqueeze(0)        # (B, N)
        score =  torch.einsum("BN, BN... -> B...", weights, dif_x_to_mu) / noise ** 2 # (B, *D)
        return score

    def _score_for_mixture_channeled(self, x, t):
        dif_x_to_mu = self.means - x.unsqueeze(1)                                   # (B, N, C, *D) = (1, N, C, *D) - (B, 1, C, *D)
        dif_x_to_mu_flat = rearrange(dif_x_to_mu, "B N C ... -> B (N C) ...")
        attn = self.attention(dif_x_to_mu_flat)  # (B, N*C)
        attn = rearrange(attn, "B (N C) -> B N C", N=self.n_modes)                  # (B, N, C)
        ##################################################
        breakpoint()
        self.log("attn", attn)
        self.log("entropy", entropy_from_mass(attn))
        self.log("max_entropy", entropy_from_mass(torch.ones_like(attn) / attn[0].numel()))
        ##################################################
        noise = self.noise(t)
        weights = attn * self.noise_gate(noise)                                     # (B, N, C)
        score =  torch.einsum("BNC, BNC... -> BC...", weights, dif_x_to_mu) / noise ** 2 # (B, C, *D)
        return score

    def _score_channeled(self, x, t):
        diff_x_to_mu = self.means - x.unsqueeze(1) # (B, N, C, D)
        attn = self.attention(diff_x_to_mu) # (B, N, C)
        weights = attn * self.noise_gate(noise).unsqueeze(0) # #(B, N, C)
        score =  torch.einsum("", weights, dif_x_to_mu) / noise ** 2 # (B, C, D)
        return score

    def extra_repr(self):
        lines = [
            f"  (score_fn): {self._score.__name__}",
            f"  (channeled): {self.channeled}",
            f"  (shp_means): {list(self.means.shape)}",
        ]
        return "\n".join(lines)

    def log(self, key, val):
        logger = globals()["LOGGER"]
        if key not in logger:
            logger[key] = []
        logger[key].append(val)


#----------------------------------------------------------------------------
# Push-Pullback Vector Field


class PushPullVF(torch.nn.Module):
    def __init__(
        self,
        vector_field,       # Vector field for most inner space 
        maps,               # Mappings from ambient space to feature space f(x, t) = z, with inv g(z, t) = x
        pullbacks,          # Operation defining how to map V(z, t) to V(x, t)
        scale=1.0,          # Scaling of the gradients
    ):
        super().__init__()
        self.vf_inner = vector_field
        self.maps = torch.nn.ModuleList(maps)
        self.pullbacks = pullbacks
        self.scale = scale

    def forward(self, x, t):
        if self.should_eval(x, t):
            return self.scale * self.encode_and_pull(x, t)
        else:
            return torch.zeros_like(x)

    def encode_and_pull(self, x, t):
        zs = self.encode(x, t)
        v_z = self.vf_inner(zs[-1], t)
        assert zs[-1].shape == v_z.shape, f"Inner z and v_z shape mismatch , got : {zs[-1].shape}, {v_z.shape}"
        v_x = self.pullback(zs, v_z, t)
        assert x.shape == v_x.shape, f"x and v_x shape mismatch, got : {x.shape}, {v_x.shape}"
        return v_x

    def encode(self, x, t):
        zs = [x]
        for map_ in self.maps:
            zs.append(map_(zs[-1], t))
        return zs

    def pullback(self, zs, v_in, noise):
        for map_, pb in zip(reversed(self.maps), reversed(self.pullbacks)):
            z_in = zs.pop()
            v_out = pb(map_.inv, z_in, v_in, noise)
            v_in = v_out 
        return v_out 

    def should_eval(self, x, t):
        if self.scale == 0:
            return False

        if hasattr(self.vf_inner, "should_eval"):
            return self.vf_inner.should_eval(x, t)
        else:
            return True

    def extra_repr(self):
        lines = [f"(scale): {self.scale} \n(pullbacks): "]
        for i, pb in enumerate(self.pullbacks):
            if hasattr(pb, "__name__"):
                name = pb.__name__
            else:
                name = type(pb).__name__
            lines.append(f"  ({i}): {name}")
        return "\n".join(lines)


#----------------------------------------------------------------------------
# A Register object is defined which maps key words to specific creation
# functions via a decorator. We build a registry for latents and pullback

class Registry:
    def __init__(self):
        self._register = {}

    def register(self, name):
        def wrapper(obj):
            self._register[name] = obj
            return obj
        return wrapper

    def __getitem__(self, key):
        return self._register[key]

registry_maps = Registry()
registry_pullback = Registry()
registry_noise = Registry()

#----------------------------------------------------------------------------
# Classes for the latents and latent inverses. 

class LatentMap(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_channeled = False
    
    def forward(self, x, *args, **kwargs):
        """Encode x to latent space"""
        raise NotImplementedError
        
    def inv(self, z, *args, **kwargs):
        """Decode z back to original space"""
        raise NotImplementedError

@registry_maps.register("ambient")
class AmbientLatentMap(LatentMap):
    def __init__(self):
        super().__init__()
        self.is_channeled = False
    
    def forward(self, x, *args, **kwargs):
        return x
    
    def inv(self, z, *args, **kwargs):
        return z

class MatrixLatentMap(LatentMap):
    def __init__(self, mat: torch.Tensor, mat_inv: torch.Tensor):
        super().__init__()
        self.register_buffer('mat', mat)
        self.register_buffer('mat_inv', mat_inv)
    
    def forward(self, x, *args, **kwargs):
        if len(x.shape) == 2:
            return torch.einsum("ld, bd -> bl", self.mat, x) 
        elif len(x.shape) == 3:
            return torch.einsum("ld, bnd -> bnl", self.mat, x) 
        else:
            raise ValueError(f"Expected tensor of rank 2 or 3 got shape : {x.shape}")
    
    def inverse(self, z, *args, **kwargs):
        if len(z.shape) == 2:
            return torch.einsum("dl, bl -> bd", self.mat_inv, z) 
        elif len(z.shape) == 3:
            return torch.einsum("dl, bnl -> bnd", self.mat_inv, z) 
        else:
            raise ValueError(f"Expected tensor of rank 2 or 3 got shape : {z.shape}")

class MatrixLatentMapChanneled(LatentMap):
    def __init__(self, mat: torch.Tensor, mat_inv: torch.Tensor):
        super().__init__()
        self.is_channeled = True
        self.register_buffer('mat', mat)
        self.register_buffer('mat_inv', mat_inv)
    
    def forward(self, x, *args, **kwargs):
        if len(x.shape) == 2:
            return torch.einsum("cld, bd -> bcl", self.mat, x) 
        elif len(x.shape) == 3:
            return torch.einsum("cld, bnd -> bncl", self.mat, x) 
        else:
            raise ValueError(f"Expected tensor of rank 2 or 3 got shape : {x.shape}")
    
    def inverse(self, z, *args, **kwargs):
        if len(z.shape) == 3:
            return torch.einsum("cdl, bcl -> bd", self.mat_inv, z) 
        elif len(z.shape) == 4:
            return torch.einsum("cdl, bncl -> bnd", self.mat_inv, z) 
        else:
            raise ValueError(f"Expected tensor of rank 3 or 4 got shape : {z.shape}")

@registry_maps.register("linear")
@registry_maps.register("linear_ch")
class RandomLinearLatentMap(LatentMap):
    def __init__(self, seed: int , dim_in: int, dim_out: int, n_features=None):
        super().__init__()

        # Create random matrix 
        g = torch.Generator().manual_seed(seed)
        if n_features:
            shp_mat = (n_features, dim_out, dim_in)
            self.is_channeled = True
            MatMap = MatrixLatentMapChanneled
        else:
            shp_mat = (dim_out, dim_in)
            MatMap = MatrixLatentMap
        # Compute pseuodo-inverse 
        mat = torch.randn(shp_mat, generator=g)
        mat_inv = torch.linalg.pinv(mat)
        self.matrix_map = MatMap(mat, mat_inv)
    
    def forward(self, x, *args, **kwargs):
        return self.matrix_map.forward(x)
    
    def inv(self, z, *args, **kwargs):
        return self.matrix_map.inverse(z)

@registry_maps.register("hf")
class HFLatentMap(LatentMap):
    def __init__(self, autoencoder, name):
        super().__init__()
        from diffusers import AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny

        Autoencoder = {
            "kl": AutoencoderKL,
            "asymmetric": AsymmetricAutoencoderKL,
            "tiny": AutoencoderTiny,
        }[autoencoder]

        try:
            self.vae = Autoencoder.from_pretrained(name, subfolder="vae", use_safetensors=True)
        except Exception:
            try:
                self.vae = Autoencoder.from_pretrained(name, use_safetensors=True)
            except Exception:
                self.vae = Autoencoder.from_pretrained(name)
        
        self.vae = self.vae.eval().requires_grad_(False)
    
    def forward(self, x, *args, **kwargs):
        return self.vae.encode(x).latent_dist.sample()
    
    def inv(self, z, *args, **kwargs):
        return self.vae.decode(z).sample
    
    def __repr__(self):
        main_str = self._get_name() + "("
        main_str += f"name : {self.vae.__class__.__name__}"
        main_str += ")"
        return main_str


@registry_maps.register("flatten")
class FlattenMap(LatentMap):
    def __init__(self):
        super().__init__()
        self.shp_og = None

    def forward(self, x, *args, **kwargs):
        self.shp_og = x.shape[1:]
        return torch.flatten(x, start_dim=1)
    
    def inv(self, z, *args, **kwargs):
        return z.view(-1, *self.shp_og)
    
#----------------------------------------------------------------------------
# Builder functions for the pullback operation
# Return a pullback function, taking input (latent_inv, x_latent, dx_latent, t)
# Not all functions require all parameters, unused parameters are marked with
# an underscore, e.g.__x_latent, __t

@registry_pullback.register("linear")
def build_pullback_linear(*args):
    def pullback_linear(latent_inv, __x_latent, dx_latent, t):
        dx = latent_inv(dx_latent, t)
        return dx
    return pullback_linear

@registry_pullback.register("numdiff")
def build_pullback_numdiff(kwargs):
    pullback = PullbackNumericalDifferentiation(**kwargs)
    return pullback

@registry_pullback.register("jvp")
def build_pullback_jvp(*args):
    @torch.no_grad
    def pullback(latent_inv, x_latent, dx_latent, t):
        _, dx = torch.autograd.functional.jvp(partial(latent_inv, t=t), x_latent, dx_latent, strict=False)
        return dx
    return pullback

#----------------------------------------------------------------------------
# Builder functions for noise and noise_dot 

@registry_noise.register("edm")
def build_noise_edm(*args):
    def noise(t) : return t
    def noise_dot(t) : return 1
    noise.args = "edm"

    return noise, noise_dot

# Functions which match a specific set of arguments to a type of object
#----------------------------------------------------------------------------

# Specification for latents
# <LatentAmbient>   := "ambient"
# <LatentLinear>    := {seed : _, dim_in: _, dim_out: _, n_features: _}
# <LatentUNetAttn>  := {net : _, attribute: "attention", index : _}
# <LatentUNetSkip>  := {net : _, attribute: "skip", index : _}
# <LatentHF>        := {"autoencoder" : _, "id": _}

def match_args_to_map(args):

    match args:
        case "ambient":
            return "ambient"
        case "flatten":
            return "flatten"
        case {"seed" : _, "dim_in" : _, "dim_out" : _, "n_features": _}:
            return "linear_ch"
        case {"seed" : _, "dim_in" : _, "dim_out" : _}:
            return "linear"
        case {"net" : _, "attribute" : "attention", "index": _}:
            return "unet-attn"
        case {"net" : _, "attribute" : "skip", "index": _}:
            return "unet-skip"
        case {"autoencoder" : _, "name" : _ }:
            return "hf"
    raise ValueError(f"Unrecognized map/map_inv args: {set(args) if isinstance(args, dict) else args!r}")

# Specification for vectorfield
# <VectorField>         := {features_template: _,  noise_gate: _, args_noise: _}
# <GuidanceVectorField> := {latent: _, vectorfield:_, noise: _}

def match_args_to_vectorfield(args):
    if not isinstance(args, dict):
        raise ValueError("args vectorfield should be dict, got type {type(args)!r}")

    required_args_gvf = set(["latent", "vectorfield", "noise"])
    required_args_vf = set(["features_template", "noise_gate", "args_noise"])

    keys = set(args)
    if required_args_gvf.issubset(keys):
        return "gvf"
    elif required_args_vf.issubset(keys):
        return "vf"
    raise ValueError(f"Unrecognized vectorfield args: {set(args)!r}")
        
# Specification for pullback
# <PullbacLinear>   := None
# <PullbackJVP>     := "jvp"
# <PullbackNumdiff> := {"step_size_slope" : _, "step_size_slope" : _}

def match_args_to_pullback(args):
    match args:
        case None:
            return "linear"
        case "jvp":
            return "jvp"
        case {"step_size_slope" : _, "step_size_intercept": _}:
            return "numdiff"
    raise ValueError(f"Unrecognized pullback args: {args!r}")

# Only accept EDM noise schedule
def match_args_to_noise(args):
    if args != "edm":
        raise ValueError

    return "edm"

#----------------------------------------------------------------------------
# Determnine if a type or args of latent is linear

def args_is_linear(args):
    return type_is_linear(match_args_to_map(args))

def type_is_linear(type_):
    return type_ in ["ambient", "linear", "linear_ch", "flatten"]
    
#----------------------------------------------------------------------------
# Builder for Push Pull Vectorfield

class BuilderPushPullVF:
    def __init__(self, args):
        self.set_args(args)

    def set_args(self, args):
        # convert to EasyDict
        self.args = dnnlib.util.to_easydict(args)

        # insert non-serializable referenced variables
        if hasattr(self.args, "references"):
            self.replace_placeholders(self.args.references)

        # ensure maps, maps_invs and pullbacks are list of EasyDicts
        for key in ("maps", "maps_invs", "pullbacks"):
            if key in self.args:
                if isinstance(self.args[key], list):
                    self.args[key] = [dnnlib.util.to_easydict(d) for d in self.args[key]]
                else:
                    self.args[key] = [self.args[key]]
    
    def replace_placeholders(self, references):
        for key, val in self.args.items():
            if key == "references":
                pass
            self.args[key] = dnnlib.util.replace_placeholders(val, references, placeholder_prefix="__REF__")

    def set_examples(self, examples):
        self.args.vector_field.means = examples

    def build(self, *, device=None):
        maps, pullbacks = self.build_maps_and_pullbacks()
        if device: maps = [map_.to(device) for map_ in maps]
        vector_field = self.build_vf(maps)
        scale = self.args.scale if hasattr(self.args, "scale") else 1.0
        ppvf = PushPullVF(vector_field, maps, pullbacks, scale)
        return ppvf

    def build_maps_and_pullbacks(self):
        maps = torch.nn.ModuleList([])
        pullbacks = []

        # Merge batch and compoennt dimension
        examples = self.args.vector_field.means
        B, N, *shp_data = examples.shape
        examples_encoded = examples.reshape(B*N, *shp_data)

        for i, args_map in enumerate(self.args.maps):
            # Handle possible none values in args.maps_invs or args.pullbacks
            args_map_inv = self.args.maps_inv[i] if hasattr(self.args, "maps_inv") else None
            args_pullback = self.args.pullbacks[i] if hasattr(self.args, "pullbacks") else None
            # Infer type of map function from args 
            type_map = match_args_to_map(args_map)
            type_map_inv = match_args_to_pullback(args_map_inv) if args_map_inv else type_map  # Copy type from type_map if no args for inverse passed

            # Infer type of pullack from args
            type_pullback = match_args_to_pullback(args_pullback)
            if type_is_linear(type_map) and (args_pullback) is not None:
                raise ValueError(f"Evaluator args can only be passed when type_map_inv is nonlinear, \n" 
                                 f"got type map : {type_map_inv} and args_pullback : {args_pullback} ")
            if not type_is_linear(type_map) and (args_pullback) is None:
                raise ValueError(f"Evaluator args must passed when type_map_inv is nonlinear, \n" 
                                 f"got type map : {type_map_inv} and args_pullback : {args_pullback} ")

            # Build maps from from registry
            map_ = registry_maps[type_map](**args_map) if isinstance(args_map, dict) else registry_maps[type_map]()
            # Build pullback from registry
            pullback = registry_pullback[type_pullback](args_pullback)
            # append to list 
            maps.append(map_)
            pullbacks.append(pullback)
            
        return maps, pullbacks
        
    def build_vf(self, maps):
        # Pick vectorfield
        VF = ScoreGatedDiracMixture
        # Construct noise gate and obtain noise function
        noise_gate = NoiseGate(**self.args.vector_field.noise_gate)
        noise, _ = registry_noise[match_args_to_noise(self.args.vector_field.noise)](self.args.vector_field.noise) 
        # Encode examples
        examples = self.args.vector_field.means
        examples_encoded = self.encode_examples(examples, maps)
        # Determine whether channeled map is contained
        num_channeled_maps = 0
        for map_ in maps:
            num_channeled_maps = map_.is_channeled
        if num_channeled_maps > 1 : 
            raise ValueError(f"Can only have one channeled map, but received {num_channeled_maps}")
        has_channeled_maps = True if num_channeled_maps > 0 else False

        vf_kwargs = {}
        if hasattr(self.args.vector_field, "kwargs"):
            vf_kwargs = self.args.vector_field.kwargs

        return VF(examples_encoded, noise_gate, noise, channeled=has_channeled_maps, **vf_kwargs)

    def encode_examples(self, examples, maps):
        # Flatten batch and examples rank
        B, N, *shp_data = examples.shape
        examples_merged = examples.reshape(B*N, *shp_data)
        # Map examples to inner space
        examples_encoded_merged = examples_merged
        for map_ in maps:
            examples_encoded_merged = map_(examples_encoded_merged, t=torch.tensor(1e-10))
        # Reshape back to original 
        examples_encoded = examples_encoded_merged.reshape(B, N, *examples_encoded_merged.shape[1:])

        return examples_encoded

