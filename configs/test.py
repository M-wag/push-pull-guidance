import torch

gradient_kwargs = {
        "scale_model_score" : 1.0, 
} 

sampler_kwargs = {
        "num_steps"         : 32, 
        "sigma_min"         : 0.002  , 
        "sigma_max"         : 80, 
        "rho"               : 7, 
        "S_churn"           : 0.0,  
        "S_min"             : 0.0, 
        "S_max"             : float('inf'), 
        "S_noise"           : 1.0,
        "dtype"             : torch.float32,
        "correct_rgb"       : False,
        "apply_2nd_order"   : True,
}



gvf_kwargs = None

generate_kwargs = {
        "ddim_inversion"        : False,
        "live_editing"          : False,
        "use_noisy_examples"    : False,
}


