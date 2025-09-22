import torch

gradient_kwargs = {
        "scale_model_score" : 1.0, 
} 

sampler_kwargs = {
        "num_steps"         : 64, 
        "sigma_min"         : 0.002  , 
        "sigma_max"         : 80, 
        "rho"               : 7, 
        # "S_churn"           : 0.0,  
        # "S_min"             : 0.0, 
        # "S_max"             : float('inf'), 
        # "S_noise"           : 1.0,
        "S_churn"           : 40.0,
        "S_min"             : 0.05, 
        "S_max"             : 50,
        "S_noise"           : 1.003, 
        "dtype"             : torch.float32,
        "correct_rgb"       : False,
        "apply_2nd_order"   : False,
}


gvf_kwargs = {
        "latent" : {"autoencoder" : "kl", "id" :"stabilityai/sd-turbo" },
        "vectorfield": {
            "features_template" : "__REF__features_template",
            "noise_gate"    : {
                "type_gate" : "heaviside", 
                "nu" : 80.0,
                "noise_onset" : 80.0,
            },
            "args_noise" : "edm",
        },
        "noise" : "edm",
        "dtype" : torch.float32,
        "scale"         : 1.0, 
        "pullback" : {"step_size_slope" : 1, "step_size_intercept": 0},
        "args_references" : {
            "features_template" : torch.zeros(1, 0, 0, 0),
        },
}

generate_kwargs = {
        "num_images"            : 10_000,
        "ddim_inversion"        : False,
        "live_editing"          : False,
        "use_noisy_examples"    : False,
}

