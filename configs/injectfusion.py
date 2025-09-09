import torch

gradient_kwargs = {
        "scale_model_score" : 1.0, 
        "gamma"             : 0.5,
        "t_edit_start"      : 32.0, 
        "h_exam_per_t"      : None
} 

sampler_args = {
        "num_steps"         : 32, 
        "sigma_min"         : 0.002  , 
        "sigma_max"         : 80, 
        "rho"               : 7, 
        "S_churn"           : 0.0,  
        "S_min"             : 0.0, 
        "S_max"             : float('inf'), 
        "S_noise"           : 1.003, 
        "dtype"             : torch.float32,
        "correct_rgb"       : False,
        "apply_2nd_order"   : True,
}


gvf_args = None

generate_kwargs = {
        "num_images"             : 10,
        "ddim_inversion"        : True,
        "live_editing"          : False,
        "use_noisy_examples"    : False,
}

