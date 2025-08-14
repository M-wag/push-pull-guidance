import torch
import dnnlib

from functools import partial
from training.networks import InjectionManager

#----------------------------------------------------------------------------
# Helper function to assign buffers for arguments that 'may' be torch.Tensor

def _maybe_register_buffer(module, name, value):
    if value is not None:
        module.register_buffer(name, torch.tensor(value))
    else:
        setattr(module, name, None)

#----------------------------------------------------------------------------
# Sigmoidal time gating function which can be either quadratic or logistic

class NoiseGate(torch.nn.Module):
    def __init__(self, type_gate: str, nu: float, decay_rate: float = None, noise_onset : float = float('inf')):
        super().__init__()
        self.type_gate = type_gate  
        self.register_buffer("nu", torch.tensor(nu))
        self.register_buffer("noise_onset", torch.tensor(noise_onset))
        _maybe_register_buffer(self, "decay_rate", decay_rate)

        if type_gate == "logistic":
            if decay_rate is None: 
                raise ValueError("decay_rate must be provided for logistic gating")
            self._gate = self._logistic_gate
        elif type_gate == "quadratic":
            self._gate = self._quadratic_gate
        else:
            raise ValueError(f"Unknown gating type: {type_gate!r}")

    def forward(self, noise):
        if noise > self.noise_onset:
            return torch.zeros_like(noise)
        return self._gate(noise)

    def _logistic_gate(self, noise):
          return torch.sigmoid(self.decay_rate * (noise - self.nu))

    def _quadratic_gate(self, noise):
        return noise**2 / (noise**2 + self.nu**2)
    
    @property
    def args(self):
        if self.type_gate == "logistic":
            return {"type_gate" : self.type_gate, "nu" : self.nu, "decay_rate" : self.decay_rate, "noise_onset" : self.noise_onset}
        else:
            return {"type_gate" : self.type_gate, "nu" : self.nu, "noise_onset" : self.noise_onset}

#----------------------------------------------------------------------------
# Pullback Operation evaluated by Numerical Differentiation

class PullbackNumericalDifferentiation(torch.nn.Module):
    def __init__(self, step_size_slope, step_size_intercept):
        super().__init__()
        self.register_buffer("a", torch.tensor(step_size_slope))
        self.register_buffer("b", torch.tensor(step_size_intercept))
    
    def step_size(self, t):
        return t * self.a + self.b

    @torch.no_grad
    def forward(self, latent_inv, x_latent, dx_latent, t):
        perturbed_latent = x_latent + self.step_size(t) * dx_latent
        f_perturbed = latent_inv(perturbed_latent, t)
        f_original = latent_inv(x_latent, t)
        dx = (f_perturbed - f_original) / self.step_size(t)
        return dx

#----------------------------------------------------------------------------
# Attention weights for the score of mixture of Gaussians
# Essentialy performs softmax(-1/2 Mahalanobis(x)^2 + ln(weight) + ln(Z))
# Includes optional modificaitons like Temperature parameter and scaling before softmax
# See equation ..

class AttentionWeightsMixture(torch.nn.Module):
    def __init__(self, means, stds, weights_mixture=None):
        super().__init__()
        self.register_buffer("means", means)    # (N, D)
        self.register_buffer("stds", stds)      # (N, )
        N ,D = means.shape
        self.D = D

        if weights_mixture is None:
            weights_mixture = torch.full((N,), 1/N, dtype=means.dtype, device=means.device)
        self.register_buffer("weights_mixture", weights_mixture)     # (N, )

        if not torch.isclose(self.weights_mixture.sum(), torch.tensor(1.0, dtype=self.weights_mixture.dtype, device=self.weights_mixture.device), atol=1e-6):
            raise ValueError(f"weights_mixture must sum to 1.0, got {self.weights_mixture.sum().item():.6f}")

    def forward(self, x, T=1.0, passing_diff=False, normalize=True):
        if passing_diff:
            diff_x_to_means = x # (B, N, D)
        else:
            diff_x_to_means = self.means.unsqueeze(0) - x.unsqueeze(1)  # (B, N, D)
            
        # Difference between x and means has variance = 1 
        if normalize:
            std_diff = diff_x_to_means.std(dim=1, unbiased=False, keepdim=True)  # shape (B, 1, D)
            diff_x_to_means = diff_x_to_means / std_diff

        mahalanobis_squared = (diff_x_to_means.pow(2).sum(-1)) / (self.stds.unsqueeze(0).pow(2)) # (B, N)
        log_weights = torch.log(self.weights_mixture + 1e-8).unsqueeze(0) # (1, N)                     
        log_partition = -self.D * torch.log(self.stds + 1e-8).unsqueeze(0) # (1, N)                    
        logits = -1/2 * mahalanobis_squared + log_weights + log_partition # (B, N)

        attention_weights = torch.nn.functional.softmax(T * logits, dim=-1)
        return attention_weights

#----------------------------------------------------------------------------
# Vectorfield of a Diffused Mixture Of Gaussians defined in lowest level of feature space

class VectorField(torch.nn.Module):
    def __init__(self, 
        features_template,              # Templates in feauture space (N, D1, D2, ....)
        noise_gate,                     # Noise-dependent sigmoidal decay function in feature space \gamma(t) -> [0 ,1]
        noise,                          # Time-depedent noise function in feature space
        noise_dot,                      # Time-depdendent derivation of noise function in feature space
        *, 
        threshold_weight    = None,     # Cut off point based on noise_gate * scale 
        threshold_time_min  = None,     # Start off point after a certain time
        threshold_time_max  = None,     # Cut off point after a cetain time
    ):
        super().__init__()
        self.register_buffer("_features_template", None)
        self.noise_gate = noise_gate
        self.noise = noise
        self.noise_dot = noise_dot
        _maybe_register_buffer(self, "threshold_weight", threshold_weight)
        _maybe_register_buffer(self, "threshold_time_min", threshold_time_min)
        _maybe_register_buffer(self, "threshold_time_max", threshold_time_max)

        self.set_features_template(features_template)
        self.setup_score()

    def forward(self, x, t):
        if self._should_apply_score(t):
            dx_guidance = self.reverse_step(x, t)
        else:
            dx_guidance = torch.zeros_like(x)
        
        return dx_guidance

    @property
    def features_template(self):
        return self._features_template 

    def set_features_template(self, templates):
        self._features_template = templates


    def reverse_step(self, x, t):
        return -self.noise_dot(t) * self.noise(t)  * self.score(x, t)

    def score(self, x, t):
        return self._score(x, t)

    def flat(self, x):
        return torch.flatten(x, start_dim=1)
    
    def _score_single_feature(self, x_latent, t):
        features_template_flat = self.features_template.squeeze(1)  # (B, 1, D) -> (B, D)
        score = self.noise_gate(self.noise(t)) * (features_template_flat - x_latent) / self.noise(t)**2
        return score

    def _score_attention(self, x_latent, t):
        diffs = self._features_template - x_latent.unsqueeze(1) #(B, N, L)
        attention = self.attention(diffs, passing_diff=True) #(B, N)
        weights = attention * self.noise_gate(self.noise(t)).unsqueeze(0) * self.scale # (N, )
        score =  torch.einsum("BN, BN... -> B...", weights, diffs) / self.noise(t) ** 2 # (B, D)
        return score

    def _should_apply_score(self, t) -> bool:
        weight = torch.max(self.noise_gate(self.noise(t)))
        apply_score = True
        # Check weight threshold
        if self.threshold_weight is not None and weight < self.threshold_weight:
            apply_score = False
        # Check time thresholds
        if self.threshold_time_min is not None and t < self.threshold_time_min:
            apply_score = False
        if self.threshold_time_max is not None and t > self.threshold_time_max:
            apply_score = False

        return apply_score 

    def attention(self, x, **kwargs):
        return self._attention_mixture(self.flat(x), **kwargs)

    def setup_score(self) -> None:
        # Check if single or multiple templates ere passed
        if self.features_template.shape[1] > 1:
            self._score = self._score_attention
            self._attention_mixture =  AttentionWeightsMixture(self.flat(self.features_template), self.noise_gate.nu)
        else:
            self.attention = None
            self._score = self._score_single_feature

    @property
    def args(self) -> dict:
        return {
            "noise_gate": self.noise_gate.args, 
            "args_noise": self.noise.args,
            "features_template" : "__REF__features_template"
        }

#----------------------------------------------------------------------------
# Encode-Pullback Vector Field

class GuidanceVF(torch.nn.Module):
    def __init__(self,
        vectorfield,        # Vectorfield of V(z,t) = v_z
        latent,             # Mapping from ambient space to feature space f(x, t) = z
        latent_inv,         # Mapping from feature space to ambient g(z, t) = x            
        pullback,           # Operation defining how to map V(z, t) to V(x, t)
        noise,              # Noise function in ambient space
        noise_dot,          # Derivative of noise function in ambient space
        scale,              # Scaling of the gradients
    ):
        super().__init__()
        self.vf_latent = vectorfield
        self.latent = latent
        self.latent_inv = latent_inv
        self._pullback = pullback
        self.noise = noise
        self.noise_dot = noise_dot
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x, t):
        if self._should_apply_score(t):
            dx_guidance = self.reverse_step(x, t)
        else:
            dx_guidance = torch.zeros_like(x)
        return dx_guidance

    @torch.no_grad
    def score(self, x, t):
        x_latent = self.latent(x, t)
        score_latent = self.vf_latent.score(x_latent, t)
        assert x_latent.shape == score_latent.shape, f"x_latent and score_latent shape mismatch , got : {x_latent.shape}, {score_latent.shape}"
        score = self._pullback(self.latent_inv, x_latent, score_latent, t)
        assert x.shape == score.shape, f"x and score shape mismatch, got : {x.shape}, {score.shape}"
        return score
    
    def reverse_step(self, x, t):
        dx = self.scale * -self.noise_dot(t) * self.noise(t) * self.score(x, t)
        return dx

    def _should_apply_score(self, t):
        return self.vf_latent._should_apply_score(t)
    
    def setup_score(self):
        self.vf_latent.setup_score
    
    @property 
    def features_template(self):
        return self.vf_latent.features_template
    
    def set_features_template(self, templates):
        # Determine datashape of template and merge templates
        B, N, *shp_data = templates.shape
        templates_merged = templates.reshape(B*N, *shp_data)
        # Calculate features of tempaltes, unmerge and save to vf_latent
        features_template_merged = self.latent(templates_merged, t=0)
        features_template = features_template_merged.reshape(B, N, *features_template_merged.shape[1:])
        self.vf_latent.set_features_template(features_template)

    @property
    def args(self):
        args_latent = self.latent._args
        args_vectorfield = self.vf_latent.args
        args_pullback = self._pullback.args
        args_noise = self.noise.args

        return {
            "latent"       : args_latent,
            "vectorfield"  : args_vectorfield,
            "pullback"     : args_pullback,
            "noise"        : args_noise,
            "scale"        : round(self.scale.item(), 3), 
        }

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

registry_latent = Registry()
registry_pullback = Registry()
registry_noise = Registry()

#----------------------------------------------------------------------------
# Builder functions for the latents and latent inverses. 
# Have input (args_latent, args_latent_inv, shape_templates, device, dtype)
# Not all build functions require all arguments.

class LatentBuilder:
    def __init__(self, args, args_inv, shp, device, dtype):
        self.args = args
        self.args_inv = args_inv
        self.shp = shp
        self.device = device
        self.dtype = dtype

    def build(self):
        latent_fn, latent_inv_fn = self._build()
        self.set_args(latent_fn)
        return latent_fn, latent_inv_fn

    def _build(self):
        raise NotImplementedError
    
    def set_args(self, fn):
        fn._args = self.args

@registry_latent.register("ambient")
class AmbientLatentBuilder(LatentBuilder):
    def _build(self):
        def latent_fn(x, t): return x
        def latent_inv_fn(z, t): return z
        return latent_fn, latent_inv_fn

class MatrixLatentBuilder(LatentBuilder):
    def _build(self):
        mat_in, mat_out = self.args, self.args_inv

        n_templates = self.shp[0]
        mat_out_stacked = torch.repeat_interleave(mat_out, dim=0, repeats=n_templates)

        def latent_fn(x):
            return torch.einsum("NOI, BI -> BNO", mat_in, x) # (batch, num_templates * num_features, dim_out)

        def latent_inv_fn(x):
            return torch.einsum("NIO, BNO -> BNI", mat_out_stacked, x) #(batch, num_templates * num_features, dim_in)

        return latent_fn, latent_inv_fn

@registry_latent.register("linear")
class RandomLinearLatentBuilder(LatentBuilder):
    def _build(self):
        g = torch.Generator(self.device).manual_seed(self.args.seed)
        shp_mat_latent = (self.args.n_features, self.args.dim_out, self.args.dim_in)
        mat_latent = torch.randn(shp_mat_latent, generator=g, device=self.device, dtype=self.dtype)
        mat_latent_inv = torch.linalg.pinv(mat_latent)

        return MatrixLatentBuilder(
                mat_latent, mat_latent_inv, self.shp, self.device, self.dtype
        )._build()

@registry_latent.register("unet")
class UNetLatentBuilder(LatentBuilder):
    def __init__(self, args, args_inv, shp, device, dtype):
        self.net = args.net
        self.index = args.index
        self.attribute = args.attribute

    def _build(self):
        self.net.set_injection_manager(InjectionManager())
        self.net.register_injection([(name, self.attribute) for name in self.names_registered])
        self.net.enable_injection_saving(True)
        self.net.enable_injection_loading(False)

        def latent_fn(x, t, *, net, attribute, names):
            # TODO : run if not ran efore
            self.last_x = x.clone.detach()
            attns = [self.net.injection_manager.load(name, self.attribute) for name in names]
            z = self.zero_padding_and_concat(attns)
            return z

        def latent_inv_fn(z, t, *, net, attribute, names):
            attns = self.undo_zero_padding_and_concat(z)
            for name, attn in zip(names, attns):
                self.net.save(name, attribute, attn)

            net.enable_injection_saving(False)
            net.enable_injection_loading(True)
            y = net(self.last_x, t)
            net.enable_injection_saving(True)
            net.enable_injection_loading(False)
            return y
        
        latent_fn = partial(latent_fn, net=self.net, names=self.names_registered, attribute=self.attribute)
        latent_inv_fn = partial(latent_inv_fn, net=self.net, names=self.names_registered, attribute=self.attribute)

        return latent_fn, latent_inv_fn

    @property
    def names_registered(self):
        if len(self.names_with_attribute) < len(self.index):
            raise ValueError(
                f"Passed more indices than layers with that attribute: "
                f"{self.names_with_attribute} (indices: {self.index})"
            )
        return [self.names_with_attribute[i] for i in self.index] 

    @property 
    def names_with_attribute(self):
        if self.attribute == "attention":
            names_with_attribute = []
            for name in self.net.names_unet_blocks["enc"]:
                if getattr(self.net.model.enc[name], "num_heads", 0) > 0:
                    names_with_attribute.append(name)
        return names_with_attribute

        
    def zero_padding_and_concat(self, tensor_list):
        """ 
        tensor_list: list of N tensors of shape [B, C_i, H_i, W_i]
        returns: single tensor [B, sum(C_i), H_max, W_max]
        """

        max_H = max(t.shape[2] for t in tensor_list)
        max_W = max(t.shape[3] for t in tensor_list)
        
        padded_tensors = []
        self.padding_metadata = {"shp_og" : []}
        
        for t in tensor_list:
            self.padding_metadata["shp_og"].append(t.shape)
            _, C, H, W = t.shape
            
            # Top-left padding (
            padding = (0, max_W - W, 0, max_H - H)
                
            padded = torch.nn.functional.pad(t, padding, mode='constant', value=0)
            padded_tensors.append(padded)
        
        return torch.cat(padded_tensors, dim=1)
                    
    def undo_zero_padding_and_concat(self, concatenated_tensor):
        shps_og = self.padding_metadata["shp_og"]
        split_tensors = torch.split(concatenated_tensor, [shp[1] for shp in shps_og ], dim=1)
        
        original_tensors = []
        for t, shape in zip(split_tensors, shps_og):
            # Simply take the top-left portion
            original_tensors.append(t[:, :, :shape[2], :shape[3]])
        return original_tensors
    
    def set_args(self, fn):
        fn._args = {"net" : "__REF__network", "attribute" : self.attribute, "index" : self.index}

@registry_latent.register("hf")
class HFLatentBuider(LatentBuilder):
    def _build(self):
        from diffusers import AutoencoderKL, AsymmetricAutoencoderKL, AutoencoderTiny
        Autoencoder = {
                "kl"        : AutoencoderKL,
                "asymmetric": AsymmetricAutoencoderKL,
                "tiny"      : AutoencoderTiny,
                }[self.args.autoencoder]

        try:
                vae = Autoencoder.from_pretrained(self.args.id, subfolder="vae", use_safetensors=True)
        except Exception:
            try:
                vae = Autoencoder.from_pretrained(self.args.id, use_safetensors=True)
            except Exception:
                vae = Autoencoder.from_pretrained(self.args.id)

        vae = vae.eval().requires_grad_(False).to(device=self.device, dtype=self.dtype)
        
        def latent_fn(x, t) : 
            return vae.encode(x).latent_dist.sample()
        def latent_inv_fn(x, t) : 
            return vae.decode(x).sample 

        return latent_fn, latent_inv_fn

#----------------------------------------------------------------------------
# Builder functions for the pullback operation
# Return a pullback function, taking input (latent_inv, x_latent, dx_latent, t)
# Not all functions require all parameters, unused parameters are marked with
# an underscore, e.g.__x_latent, __t

@registry_pullback.register("linear")
def build_pullback_linear(*args):
    def pullback(latent_inv, __x_latent, dx_latent, t):
        dx = latent_inv(dx_latent, t)
        return dx
    pullback.args = args[0]
    return pullback

@registry_pullback.register("numdiff")
def build_pullback_numdiff(kwargs):
    pullback = PullbackNumericalDifferentiation(**kwargs)
    pullback.args = kwargs
    return pullback

@registry_pullback.register("jvp")
def build_pullback_jvp(*args):
    @torch.no_grad
    def pullback(latent_inv, x_latent, dx_latent, t):
        _, dx = torch.autograd.functional.jvp(partial(latent_inv, t=t), x_latent, dx_latent, strict=False)
        return dx
    pullback.args = "jvp" 
    return pullback

#----------------------------------------------------------------------------
# Builder functions for noise and noise_dot 

@registry_noise.register("edm")
def buld_noise_edm(*args):
    def noise(t) : return t
    def noise_dot(t) : return 1
    noise.args = "edm"

    return noise, noise_dot

# Functions which match a specific set of arguments to a type of object
#----------------------------------------------------------------------------

# Specification for latents
# <LatentAmbient>   := "ambient"
# <LatentLinear>    := {seed : _, dim_in: _, dim_out: _, n_features: _}
# <LatentUNet>      := {net : _, attribute: _, index : _}
# <LatentHF>        := {"autoencoder" : _, "id": _}

def match_args_to_latent(args):
    match args:
        case "ambient":
            return "ambient"
        case {"seed" : _, "dim_in" : _, "dim_out" : _, "n_features": _}:
            return "linear"
        case {"net" : _, "attribute" : _, "index": _}:
            return "unet"
        case {"autoencoder" : _, "id" : _ }:
            return "hf"
    raise ValueError(f"Unrecognized latent/latent_inv args: {set(args) if isinstance(args, dict) else args!r}")

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
    return type_is_linear(match_args_to_latent(args))

def type_is_linear(type_):
    return type_ in ["ambient", "linear"]
    
#----------------------------------------------------------------------------
# Allow create_gvf to take different kwargs by name
# and ensure that they are valid GuidanceVectorfield arguments

def create_gvf(
        latent,            # Arguments for latent function ("latent")
        vectorfield,       # Arguments for vectorfield in feature space ("vectorfield")
        noise,             # Arguments for time-dependent noise function used during reverse step ("noise")
        pullback   = None, # Arguments for the pullback operation ("pullback")
        latent_inv = None, # Arguments for latent (pseudo)-inverse function ("latent_inv")
        **kwargs,
    ):

    if match_args_to_vectorfield( {"latent" : latent, "vectorfield": vectorfield, "noise" : noise}) != "gvf":
         raise ValueError("Arguments do not match those for a GuidaneVectorField")

    return _create_gvf(latent, vectorfield, noise, pullback, latent_inv, **kwargs)

#----------------------------------------------------------------------------
# Creates an object of GuidanceVF based on a nested dictionary of args
def _create_gvf(
        args_latent,            # Arguments for latent function ("latent")
        args_vectorfield,       # Arguments for vectorfield in feature space ("vectorfield")
        args_noise,             # Arguments for time-dependent noise function used during reverse step ("noise")
        args_pullback   = None, # Arguments for the pullback operation ("pullback")
        args_latent_inv = None, # Arguments for latent (pseudo)-inverse function ("latent_inv")
        args_references = {},   # Arguments which are not serializable and are passed by reference
        scale           = 1.0,  # Scale of the gvf guidance
        device          = "cuda" if torch.cuda.is_available() else "cpu",
        dtype           = torch.float32,
    ):

    # Insert non-serializable referenced variables
    args_latent, args_vectorfield, args_noise, args_pullback, args_latent_inv = [
        dnnlib.util.replace_placeholders(x, args_references, placeholder_prefix="__REF__") 
        for x in (args_latent, args_vectorfield, args_noise, args_pullback, args_latent_inv)
    ]

    # Wrap args into EasyDict if they're plain dicts
    args_latent, args_vectorfield, args_noise, args_pullback, args_latent_inv = [
        dnnlib.util.to_easydict(x) for x in
        (args_latent, args_vectorfield, args_noise, args_pullback, args_latent_inv)
    ]

    # Infer type of latent function from args and build from registry.
    type_latent = match_args_to_latent(args_latent)
    type_latent_inv = match_args_to_pullback(args_latent_inv) if args_latent_inv else type_latent  # Copy type from type_latent if no args for inverse passed
    shp_templates = args_vectorfield.features_template.shape # Get template shape, necessary for initiliazation of some latents
    latent_fn, latent_inv_fn = registry_latent[type_latent](args_latent, args_latent_inv, shp_templates, device, dtype).build()

    # Infer type of pullback and build from registry
    type_pullback = match_args_to_pullback(args_pullback)
    if type_is_linear(type_latent) and (args_pullback) is not None:
        raise ValueError(f"Evaluator args can only be passed when type_latent_inv is nonlinear, \n" 
                         f"got type latent : {type_latent_inv} and args_pullback : {args_pullback} ")

    pullback = registry_pullback[type_pullback](args_pullback)

    # Create vectorfield 
    type_vf = match_args_to_vectorfield(args_vectorfield)
    if type_vf == "gvf":
        vectorfield = create_gvf(**args_vectorfield, args_references=args_references)
    else:
        noise_latent, noise_dot_latent = registry_noise[match_args_to_noise(args_vectorfield.args_noise)](args_vectorfield.args_noise)  # latent noise
        args_vectorfield.noise_gate = NoiseGate(**args_vectorfield.noise_gate) # latent noise gate
        del args_vectorfield["args_noise"]
        args_vectorfield.noise = noise_latent
        args_vectorfield.noise_dot = noise_dot_latent
        vectorfield = VectorField(**args_vectorfield)
    
    # Get noise function
    noise, noise_dot = registry_noise[match_args_to_noise(args_noise)](args_noise) 

    gvf = GuidanceVF(vectorfield, latent_fn, latent_inv_fn, pullback, noise, noise_dot, scale)
    return gvf


