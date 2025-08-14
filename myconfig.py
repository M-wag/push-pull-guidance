import torch

sampler_args = {
        "scale_model_score" : 1.0,
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
        # "latent" : {"autoencoder" : "kl", "id" :"stabilityai/sd-turbo" },
        "latent" : {"net" : "__REF__network", "attribute" :"attention", "index" : [-3, -2, -1] },

        "vectorfield": {
            "features_template" : "__REF__features_template",
            "noise_gate"    : {
                "type_gate" : "logistic", 
                "nu" : 10.0,
                "decay_rate" : 2.0,
                "noise_onset" : 70.0,
            },
            "args_noise" : "edm",
        },
        "noise" : "edm",
        "dtype" : torch.float32,
        "scale"         : 1.0, 
        "pullback" : {"step_size_slope" : 0.01, "step_size_intercept": 0},
        "args_references" : {
            "features_template" : torch.zeros(1, 0, 0, 0),
        },
}


