import torch

determinstic_sampling = {
        "S_churn"           : 0.0,  
        "S_min"             : 0.0, 
        "S_max"             : float('inf'), 
}

stochastic_sampling = {
        "S_churn"           : 40.0,
        "S_min"             : 0.05, 
        "S_max"             : 50,
}

first_order = {
        "num_steps"         : 64,
        "apply_2nd_order"   : False,
}

second_order = {
        "num_steps"         : 32,
        "apply_2nd_order"   : True,
}

sampler_kwargs = {
        "sigma_min"         : 0.002 , 
        "sigma_max"         : 80, 
        "rho"               : 7, 
        "S_noise"           : 1.0,
        **determinstic_sampling,
        "noise_seed"        : 0,
        "dtype"             : torch.float32,
        "correct_rgb"       : False,
        **second_order, 
}

gvf_kwargs = {
        "scale"         : 1.0,
        "maps"          : [
                            {"autoencoder" : "kl", "name" :"stabilityai/sd-turbo" },
                            "flatten",
                            {"seed": 0, "dim_in" : 256, "dim_out" : 128, "n_features" : 3},
                           ],
        "vector_field"  : {
            "noise_gate"    : { "type_gate" : "heaviside", "nu" : 5.5, },
            "noise"         : "edm",
            },
        "pullbacks"     : [
                            {"step_size_slope" : 1, "step_size_intercept": 0},
                            None,
                            None,
                           ],
    }

gvf_kwargs = None


generate_kwargs = {
        "ddim_inversion"        : False,
        "live_editing"          : False,
        "use_noisy_examples"    : False,
}

gradient_kwargs = {
        "scale_model_score" : 1.0, 
} 
