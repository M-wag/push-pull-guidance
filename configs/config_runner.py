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
        **stochastic_sampling,
        "dtype"             : torch.float32,
        "correct_rgb"       : False,
        **second_order, 
}


gvf_kwargs = None

generate_kwargs = {
        "ddim_inversion"        : False,
        "live_editing"          : False,
        "use_noisy_examples"    : False,
        "example_idx_range"     : None,
        "class_idx"             : [],
}

gradient_kwargs = {
        "scale_model_score" : 1.0, 
} 
