import torch

gradient_kwargs = {
        "scale_model_score" : 1.0, 
} 
sampler_args = {
        "num_steps"         : 32, 
        "sigma_min"         : 0.002  , 
        "sigma_max"         : 80, 
        "rho"               : 7, 
        "S_churn"           : 0.0,  
        "S_min"             : 0.0, 
        "S_max"             : float('inf'), 
        "S_noise"           : 1, 
        "dtype"             : torch.float32,
        "correct_rgb"       : False,
        "apply_2nd_order"   : True,
}


gvf_args = {
        # "latent" : "ambient",
        # "latent" : {"autoencoder" : "asymmetric", "id" :"cross-attention/asymmetric-autoencoder-kl-x-1-5" },
        "latent" : {"autoencoder" : "kl", "id" :"stabilityai/sd-turbo" },
        # "latent" : {"net" : "__REF__network", "attribute" :"attention", "index" : list(range(4,9)) },

        "vectorfield": {
            "features_template" : "__REF__features_template",
            "noise_gate"    : {
                "type_gate" : "quadratic", 
                "nu" : 6.5,
                # "decay_rate" : 100.,
                "noise_onset" :  80.0,
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


