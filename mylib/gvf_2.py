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
# Vectorfield of a Mixture Of Gaussians defined in lowest level of feature space

class VectorField:
    def __init__(self, 
        features_template,              # Templates in feauture space
        scale,                          # Scaling of score in feature space
        noise_gate,                     # Noise-dependent sigmoidal decay function in feature space \gamma(t) -> [0 ,1]
        noise,                          # Time-depedent noise function in feature space
        noise_dot                       # Time-depdendent derivation of noise function in feature space
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
        self.noise = lambda x: x 
        self.noise_dot = lambda x : 1 

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
# Functions which match a specific set of arguments to a type of object

def match_args_to_latent(args):
    match args:
        case "ambient":
            return "ambient"
        case {"step_size" : _}:
            return "numdiff"
        case {"hf_url" : _}:
            return "hf"
        case {"net" : _, "hook_manager" : _}:
            return "unet"
    raise ValueError(f"Unrecognized latent/latent_inv args: {args!r}")

def match_args_to_vectorfield(args):
    if not isinstance(args, dict):
        raise ValueError("args vectorfield should be dict, got type {type(args)!r}")

    required_args_gvf = set(["latent", "vectorfield", "noise", "noise_dot"])
    required_args_vf = set(["features_template", "scale", "noise_gate", "noise", "noise_dot"])

    keys = set(args)
    if required_args_gvf.issubset(keys):
        return "gvf"
    elif required_args_vf.issubset(keys):
        return "vf"
    raise ValueError(f"Unrecognized vectorfield args: {args!r}")
        

def match_args_to_pullback(args):
    match args:
        case None:
            return "linear"
        case "jvp":
            return "jvp"
        case {"step_size" : _}:
            return "numdiff"
    raise ValueError(f"Unrecognized pullback args: {args!r}")

def match_args_to_noise(args):
    if args is not "edm":
        raise ValueError
    noise = lambda t : t
    noise_dot = lambda t : t

    return noise, noise_dot

#----------------------------------------------------------------------------
# Determnine if a type of latent is linear

def is_linear(type_latent):
    return type_latent in ["ambient", "linear"]
    
#----------------------------------------------------------------------------
# Creates an object of GuidanceVF based on a nested dictionary of args

def create_gvf(
    args_latent,            # Arguments for latent function ("latent")
    args_vectorfield,       # Arguments for vectorfield in feature space ("vectorfield")
    args_noise,             # Arguments for time-dependent noise function used during reverse step ("noise")
    args_pullback   = None  # Arguments for the pullback operation ("pullback")
    args_latent_inv = None  # Arguments for latent (pseudo)-inverse function ("latent_inv")
):

    # Infer type of latent function from args and build from registry.
    type_latent = match_args_to_latent(args_latent)
    type_latent_inv = match_args_to_pullback(args_latent_inv) if args_latent_inv else type_latent  # Copy type from type_latent if no args for inverse passed
    latent_fn, latent_inv_fn = registry_latent[type_latent](args_latent, args_latent_inv)

    # Infer type of pullback and build from registry
    type_pullback = match_args_to_pullback(args_pullback)
    if is_linear(type_latent)  and (args_evaluator) is not None:
        raise ValueError(f"Evaluator args can only be passed when type_latent_inv is nonlinear, \n" 
                         f"got type latent : {type_latent_inv} and args_evaluator : args_evaluator} ")

    pullback = registry_pulllback[type_pullbac](args_pullback)

    # Create vectorfield 
    type_vf = match_args_to_vf(args_vectorfield)
    if type_vf is "gvf":
        args_vectorfield["noise_gating"] = NoiseGate(args_vectorfield["noise_gating"]) # initialize NoiseGate 
        vectorfield = create_gvf(args_vectorfield)
    else:
        vectorfield = VectorField(args_vectorfield)
    
    # Get noise function
    noise, noise_dot = match_args_to_latent(args_noise)

    gvf = GuidanceVF(vectorfield, latent, latent_inv, noise, noise_dot)
    return gvf


