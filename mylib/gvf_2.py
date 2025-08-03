import torch
import dnnlib

# Sigmoidal time gating function which can be either quadratic or logistic

class NoiseGate:
    def __init__(self, type_gate: str, nu: float, decay_rate: float = None):
        self.nu = nu
        self.decay_rate = decay_rate
        
        # TODO: This is ugly, please refactor
        if type_gate == "logistic":
            self.args = {"type_gate" : type_gate, "nu" : nu, "decay_rate" : decay_rate}
        else:
            self.args = {"type_gate" : type_gate, "nu" : nu}

        if type_gate == "logistic":
            if decay_rate is None: 
                raise ValueError("decay_rate must be provided for logistic gating")
            self._fn = self._logistic_gate
        elif type_gate == "quadratic":
            self._fn = self._quadratic_gate
        else:
            raise ValueError(f"Unknown gating type: {type_gate!r}")

    def __call__(self, noise):
        return self._fn(noise)

    def _logistic_gate(self, noise):
          return torch.sigmoid(self.decay_rate * (noise - self.nu))

    def _quadratic_gate(self, noise):
        return noise**2 / (noise**2 + self.nu**2)
#----------------------------------------------------------------------------
# Pullback Operation evaluated by Numerical Differentiation

class PullbackNumericalDifferentiation():
    def __init__(self, step_size_slope, step_size_intercept):
        self.a = step_size_slope
        self.b = step_size_intercept
    
    def step_size(self, t):
        return t * self.a + self.b

    @torch.no_grad
    def __call__(self, latent_inv, x_latent, dx_latent, t):
        perturbed_latent = x_latent + self.step_size(t) * dx_latent
        f_perturbed = latent_inv(perturbed_latent)
        f_original = latent_inv(x_latent)
        dx = (f_perturbed - f_original) / self.step_size(t)
        return dx

#----------------------------------------------------------------------------
# Attention weights for the score of mixture of Gaussians
# Essentialy performs softmax(-1/2 Mahalanobis(x)^2 + ln(weight) + ln(Z))
# Includes optional modificaitons like Temperature parameter and scaling before softmax
# See equation ..

class AttentionWeightsMixture:
    def __init__(self, means, stds, weights_mixture=None):
        self.means = means #(N, D)
        self.stds  = stds #(N,)
        N, D = means.shape
        self.D = D

        if weights_mixture is None:
            weights_mixture = torch.full((N,), 1/N, dtype=means.dtype, device=means.device)
        self.weights_mixture = weights_mixture #(N, )

        if not torch.isclose(self.weights_mixture.sum(), torch.tensor(1.0, dtype=self.weights_mixture.dtype, device=self.weights_mixture.device), atol=1e-6):
            raise ValueError(f"weights_mixture must sum to 1.0, got {self.weights_mixture.sum().item():.6f}")

    def __call__(self, x, T=1.0, passing_diff=False, normalize=True):
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

        attention_weights = torch.nn.F.softmax(T * logits, dim=-1)
        return attention_weights

#----------------------------------------------------------------------------
# Vectorfield of a Diffused Mixture Of Gaussians defined in lowest level of feature space

class VectorField:
    def __init__(self, 
        features_template,              # Templates in feauture space (N, D1, D2, ....)
        scale,                          # Scaling of score in feature space
        noise_gate,                     # Noise-dependent sigmoidal decay function in feature space \gamma(t) -> [0 ,1]
        noise,                          # Time-depedent noise function in feature space
        noise_dot,                      # Time-depdendent derivation of noise function in feature space
        *, 
        flatten_input       = False,    # Whether to flatten input [..., C, H, W] -> [..., (C H W)]
        threshold_weight    = None,     # Cut off point based on noise_gate * scale 
        threshold_time_min  = None,     # Start off point after a certain time
        threshold_time_max  = None,     # Cut off point after a cetain time
    ):

        self.flatten_input = flatten_input
        self.scale = scale
        self.noise_gate = noise_gate
        self.noise = noise
        self.noise_dot = noise_dot
        self.threshold_weight = threshold_weight
        self.threshold_time_min = threshold_time_min
        self.threshold_time_max = threshold_time_max
        self.features_template = features_template
        self.setup_score()

    def __call__(self, x, t):
        if self.flatten_input:
            x = self.flat(x) 
        
        if self._should_apply_score(t):
            dx_guidance = self.reverse_step(x, t)
        else:
            dx_guidance = torch.zeros_like(x)
        
        if self.flatten_input:
            dx_guidance = self.unflat(dx_guidance) 

        return dx_guidance

    @property
    def features_template(self):
        return self._features_template 

    @features_template.setter
    def features_template(self, x):
        self._features_template = self.flat(x) if self.flatten_input else x
        # If templates is a Tensor save shape
        if isinstance(self._features_template, torch.Tensor):
            self._shp_og = self._features_template.shape[1:]
        else: 
            self.shp_og = None

    def reverse_step(self, x, t):
        return -self.noise_dot(t) * self.noise(t)  * self.score(x, t)

    def score(self, x, t):
        return self._score(x, t)

    def flat(self, x):
        return torch.flatten(x, start_dim=1)
    
    def unflat(self, x):
        return torch.unflatten(x, start_dim=1, sizes=self._shp_og)

    def _score_single_feature(self, x_latent, t):
        score = self.scale * self.noise_gate(self.noise(t)) * (self._features_template - x_latent) / self.noise(t)**2
        return score

    def _score_attention(self, x_latent, t):
        diffs = self._features_template - x_latent.unsqueeze(1) #(B, N, D)
        attention = self.attention(diffs) #(B, N)
        weights = attention * self.noise_gate(noise(t)).unsqueeze(0) * self.scale # (N, )
        score =  torch.einsum("BN, BN... -> B...", weights, diffs) / self.noise(t) ** 2 # (B, D)
        return score

    def _should_apply_score(self, t) -> bool:
        weight = torch.max(self.noise_gate(self.noise(t)) * self.scale)
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

    def setup_score(self) -> None:
        if len(self._shp_og) < 2:
            raise ValueError(f"Features template should be at least rank 2 / shape (N, D), got :{self._shp_og}")
        # Check if single or multiple templates ere passed
        if self.features_template.shape[0] > 1:
            self._score = self._score_attention
            self.attention = lambda x : AttentionWeightsMixture(features_template, self.noise_gate.nu,)(self.flatten(x))
        else:
            self.attention = None
            self._score = self._score_single_feature

    @property
    def args(self) -> dict:
        return {
            "noise_gate": self.noise_gate.args, 
            "args_noise": self.noise.args,
            "scale": self.scale,
            "features_template" : "__REF__features_template"
        }

#----------------------------------------------------------------------------
# Encode-Pullback Vector Field

class GuidanceVF:
    def __init__(self,
        vectorfield,        # Vectorfield of V(z,t) = v_z
        latent,             # Mapping from ambient space to feature space f(x, t) = z
        latent_inv,         # Mapping from feature space to ambient g(z, t) = x            
        pullback,           # Operation defining how to map V(z, t) to V(x, t)
        noise,              # Noise function in ambient space
        noise_dot,          # Derivative of noise function in ambient space
    ):

        self.vf_latent = vectorfield
        self.latent = latent
        self.latent_inv = latent_inv
        self._pullback = pullback
        self.noise = noise
        self.noise_dot = noise_dot

    def __call__(self, x, t):
        if self._should_apply_score(t):
            dx_guidance = self.reverse_step(x, t)
        else:
            dx_guidance = torch.zeros_like(x)
        return dx_guidance

    @torch.no_grad
    def score(self, x, t):
        x_latent = self.latent(x, t)
        score_latent = self.vf_latent.score(x_latent, t)
        score = self._pullback(x_latent, score_latent, t)
        return score
    
    def reverse_step(self, x, t):
        dx = -self.noise_dot(t) * self.noise(t) * self.score(x, t)
        return dx

    def _should_apply_score(self, t):
        return self.vf_latent._should_apply_score(t)
    
    def setup_score(self):
        self.vf_latent.setup_score
    
    @property
    def args(self):
        args_latent = self.latent.args
        args_vectorfield = self.vf_latent.args
        args_pullback = self._pullback.args
        args_noise = self.noise.args

        return {
            "latent"       : args_latent,
            "vectorfield"  : args_vectorfield,
            "pullback"     : args_pullback,
            "noise"        : args_noise,
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

@registry_latent.register("ambient")
def build_latent_ambient(*args):
    def latent_fn(x, t): return x
    def latent_inv_fn(x):return x
    latent_fn.args = args[0]
    return latent_fn, latent_inv_fn

def build_latent_from_matrix(mat_in, mat_out, shp_templates, device, dtype):
    def latent_fn(x): 
        return torch.einsum("NOI, BI -> BNO", mat_in, x) # (batch, num_templates * num_features, dim_out)

    n_templates = shp_templates[0]
    mat_out_stacked = torch.repeat_interleave(mat_out, dim=0, repeats=n_templates)
    def latent_inv_fn(x):
        return torch.einsum("NIO, BNO -> BNI", mat_out_stacked, x) #(batch, num_templates * num_features, dim_in)

    return latent_fn, latent_inv_fn

@registry_latent.register("linear")
def build_latent_random_linear(args, _, shp_templates, device, dtype):
    # Construct the matrix from seed, along with pseudoinverse
    g = torch.Generator(device).manual_seed(args.seed)
    shp_mat_latent = (args.n_features, args.dim_out, args.dim_in)
    mat_latent = torch.randn(shp_mat_latent, generator=g, device=device, dtype=dtype)
    mat_latent_inv = torch.linalg.pinv(mat_latent)
    latent_fn, latent_inv_fn = build_latent_from_matrix(mat_latent, mat_latent_inv, shp_templates, device, dtype)
    latent_fn.args = args
    return latent_fn, latent_inv_fn

@registry_latent.register("unet")
def build_latent_unet(args, _args_out, _shp, device, dtype):
    def latent_fn(x, t): return x
    def latent_inv_fn(x):return x
    latent_fn.args = {"net" : "__REF__network", "hook_manager": "__REF__hook_manager"}
    return latent_fn, latent_inv_fn


#----------------------------------------------------------------------------
# Builder functions for the pullback operation
# Return a pullback function, taking input (latent_inv, x_latent, dx_latent, t)
# Not all functions require all parameters, unused parameters are marked with
# an underscore, e.g.__x_latent, __t

@registry_pullback.register("linear")
def build_pullback_linear(*args):
    def pullback(latent_inv, __x_latent, dx_latent, __t):
        dx = latent_inv(dx_latent)
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
    def pullback(latent_inv, x_latent, dx_latent, __t):
        _, dx = torch.autograd.functional.jvp(latent_inv, x_latent, dx_latent, strict=False)
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
# <LatentHF>        := {hf_url : _}
# <LatentUNet>      := {net : _, hook_manager: _}

def match_args_to_latent(args):
    match args:
        case "ambient":
            return "ambient"
        case {"seed" : _, "dim_in" : _, "dim_out" : _, "n_features": _}:
            return "linear"
        case {"hf_url" : _}:
            return "hf"
        case {"net" : _, "hook_manager" : _}:
            return "unet"
    raise ValueError(f"Unrecognized latent/latent_inv args: {set(args) if isinstance(args, dict) else args!r}")

# Specification for vectorfield
# <VectorField>         := {features_template: _, scale: _, noise_gate: _, args_noise: _}
# <GuidanceVectorField> := {latent: _, vectorfield:_, noise: _}

def match_args_to_vectorfield(args):
    if not isinstance(args, dict):
        raise ValueError("args vectorfield should be dict, got type {type(args)!r}")

    required_args_gvf = set(["latent", "vectorfield", "noise"])
    required_args_vf = set(["features_template", "scale", "noise_gate", "args_noise"])

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
        device          = "cuda" if torch.cuda.is_available() else "cpu",
        dtype           = torch.float64,
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
    latent_fn, latent_inv_fn = registry_latent[type_latent](args_latent, args_latent_inv, shp_templates, device, dtype)

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

    gvf = GuidanceVF(vectorfield, latent_fn, latent_inv_fn, pullback, noise, noise_dot)
    return gvf


