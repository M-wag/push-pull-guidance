import torch
import dnnlib

#----------------------------------------------------------------------------
# Sigmoidal time gating function which can be either quadratic or logistic

class NoiseGate:
    def __init__(self, type_gate: str, nu: float, decay_rate: float = None):
        self.nu = nu
        self.decay_rate = decay_rate

        if type_gate == "logistic":
            if decay_rate is None: raise ValueError("decay_rate must be provided for logistic gating")
            self._fn = lambda t: torch.sigmoid(self.decay_rate * (noise - self.nu)) 
        elif type_gate == "quadratic":
            self._fn = lambda noise: noise**2 / (noise**2 + self.nu**2)
        else:
            raise ValueError(f"Unknown gating type: {type_gate!r}")

    def __call__(self, noise):
        return self._fn(noise)

#----------------------------------------------------------------------------
# Pullback Operation evaluated by Numerical Differentiation

class PullbackNumericalDifferentiation():
    def __init__(self, step_size: callable):
        self.step_size = step_size  

    @torch.no_grad
    def __call__(latent_inv, x_latent, dx_latent, t):
        perturbed_latent = x_latent + self.step_size(t) * dx_latent
        f_perturbed = latent_inv(perturbed_latent)
        f_original = latent_inv(x_latent)
        dx = (f_perturbed - f_original) / self.step_size(t)
        return dx

#----------------------------------------------------------------------------
# Vectorfield of a Mixture Of Gaussians defined in lowest level of feature space

class VectorField:
    def __init__(self, 
        features_template,              # Templates in feauture space
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

        # Core parameters
        self.features_template = self.flat(features_template) if flatten_input else features_template
        self.scale = scale
        self.noise_gate = noise_gate
        self.noise = noise
        self.noise_dot = noise_dot

        self.time_weight = lambda t: torch.sigmoid(self.decay_rate * (self.noise(t) - self.v_0)) 
        # Optional features
        self.flatten_input = flatten_input
        self.threshold_weight = threshold_weight
        self.threshold_time_min = threshold_time_min
        self.threshold_time_max = threshold_time_max

         # TODO : Not sure how this is gonna be implemented, now done by Builder
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
        self.device = self.features_template.device
        self.dtype = self.features_template.dtype
        

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

    def flat(self, x):
        return rearrange(x, "... c h w -> ... (c h w)")
    
    def unflat(self, x):
        return rearrange(x, "... (c h w) -> ... c h w", c=self.templates.shape[-3], h=self.templates.shape[-2], w=self.templates.shape[-1])

    def _score_single_feature(self, x_latent, t):
        score = self.scale * self.time_weight(t) * (self.features_template - x_latent) / self.noise(t)**2
        return score

    def _score_attention(self, x_latent, t):
        diffs = self.features_template - x.unsqueeze(1) #(B, N, D)
        # TODO : normalizer not init
        # TODO : can you normlaize for mixture attention
        diffs_normalized = self.attention_normalizer(diffs) #(B, N, D)
        attention = self.attention(diffs_normalized) #(B, N)
        weights = attention * self.time_weight.unsqueeze(0) * self.scale # (N, )
        score =  torch.einsum("BN, BN... -> B...", weights, recons) / self.noise(t) ** 2 # (B, D)
        return score

#----------------------------------------------------------------------------
# Encode-Pullback Vector Field

class GuidanceVF():
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
        x_latent = self.latent(x)
        score_latent = self.vf_latent.score(x_latent, t)
        score = self._pullback(x_latent, score_latent, t)
        return score
    
    def reverse_step(self, x, t):
        dx = -self.noise_dot(t) * self.noise(t) * self.score(x, t)

    def _should_apply_score(self, t):
        return self.vf_latent.should_apply_score(t)

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

#----------------------------------------------------------------------------
# Builder functions for the latents and latent inverses. 
# Have input (args_latent, args_latent_inv, shape_templates, device, dtype)
# Not all build functions require all arguments.

@registry_latent.register("ambient")
def build_latent_ambient(*args):
    return lambda x : x, lambda x : x

def build_latent_from_matrix(mat_in, mat_out, shp_templates, device, dtype):
    def latent_fn(x, t): 
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
    return build_latent_from_matrix(mat_latent, mat_latent_inv, shp_templates, device, dtype)

#----------------------------------------------------------------------------
# Builder functions for the pullback operation
# Return a pullback function, taking input (latent_inv, x_latent, dx_latent, t)
# Not all functions require all parameters, unused parameters are marked with
# an underscore, e.g.__x_latent, __t

@registry_pullback.register("linear")
def build_jvp(*args):
    def pullback(latent_inv, __x_latent, dx_latent, __t):
        dx = latent_inv(dx_latent)
        return dx
    return pullback

@registry_pullback.register("numdiff")
def build_jvp(step_size):
    pullback = PullbackNumericalDifferentiation(step_size)
    return pullback

@registry_pullback.register("jvp")
def build_jvp(*args):
    @torch.no_grad
    def pullback(latent_inv, __x_latent, dx_latent, __t):
        _, dx = torch.autograd.functional.jvp(latent_inv, x_latent, dx_latent, strict=False)
        return dx
    return pullback

#----------------------------------------------------------------------------
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

#----------------------------------------------------------------------------
# Specification for vectorfield
# <VectorField>         := {features_template: _, scale: _, noise_gate: _, args_noise: _}
# <GuidanceVectorField> := {latent: _, vectorfield:_, noise: _, noise_dot}

def match_args_to_vectorfield(args):
    if not isinstance(args, dict):
        raise ValueError("args vectorfield should be dict, got type {type(args)!r}")

    required_args_gvf = set(["latent", "vectorfield", "noise", "noise_dot"])
    required_args_vf = set(["features_template", "scale", "noise_gate", "args_noise"])

    keys = set(args)
    if required_args_gvf.issubset(keys):
        return "gvf"
    elif required_args_vf.issubset(keys):
        return "vf"
    raise ValueError(f"Unrecognized vectorfield args: {set(args)!r}")
        
#----------------------------------------------------------------------------
# Specification for vectorfield
# <PullbacLinear>   := None
# <PullbackJVP>     := "jvp"
# <PullbackNumdiff> := {latent: _, vectorfield:_, noise: _, noise_dot}

def match_args_to_pullback(args):
    match args:
        case None:
            return "linear"
        case "jvp":
            return "jvp"
        case {"step_size" : _}:
            return "numdiff"
    raise ValueError(f"Unrecognized pullback args: {args!r}")

#----------------------------------------------------------------------------
# Only accept EDM noise schedule
def match_args_to_noise(args):
    if args != "edm":
        raise ValueError
    noise = lambda t : t
    noise_dot = lambda t : 1

    return noise, noise_dot

#----------------------------------------------------------------------------
# Determnine if a type or args of latent is linear

def args_is_linear(args):
    return type_is_linear(match_args_to_latent(args))

def type_is_linear(type_):
    return type_ in ["ambient", "linear"]
    
#----------------------------------------------------------------------------
# Creates an object of GuidanceVF based on a nested dictionary of args

def create_gvf(
    args_latent,            # Arguments for latent function ("latent")
    args_vectorfield,       # Arguments for vectorfield in feature space ("vectorfield")
    args_noise,             # Arguments for time-dependent noise function used during reverse step ("noise")
    args_pullback   = None, # Arguments for the pullback operation ("pullback")
    args_latent_inv = None, # Arguments for latent (pseudo)-inverse function ("latent_inv")
    device          = "cuda" if torch.cuda.is_available() else "cpu",
    dtype           = torch.float64,
):

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
    if type_is_linear(type_latent)  and (args_pullback) is not None:
        raise ValueError(f"Evaluator args can only be passed when type_latent_inv is nonlinear, \n" 
                         f"got type latent : {type_latent_inv} and args_pullback : {args_pullback} ")

    pullback = registry_pullback[type_pullback](args_pullback)

    # Create vectorfield 
    type_vf = match_args_to_vectorfield(args_vectorfield)
    if type_vf == "gvf":
        vectorfield = create_gvf(**args_vectorfield)
    else:
        noise_latent, noise_dot_latent = match_args_to_noise(args_vectorfield.args_noise) # initialize latent noise
        print(args_vectorfield.noise_gate)
        args_vectorfield.noise_gate = NoiseGate(**args_vectorfield.noise_gate) # initialize NoiseGate 
        del args_vectorfield["args_noise"]
        args_vectorfield.noise = noise_latent
        args_vectorfield.noise_dot = noise_dot_latent
        vectorfield = VectorField(**args_vectorfield)
    
    # Get noise function
    noise, noise_dot = match_args_to_latent(args_noise)

    gvf = GuidanceVF(vectorfield, latent, latent_inv, noise, noise_dot)
    return gvf


