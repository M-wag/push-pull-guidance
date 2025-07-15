import calculate_metrics

if __name__ == "__main__":
    # Deterministic EDM sampler kwargs
    sampler_kwargs = {
            "num_steps"   : 16, 
            "sigma_min"   : 0.002  , 
            "sigma_max"   : 80, 
            "rho"         : 7, 
            "S_churn"     : 0.0,  
            "S_min"       : 0.0, 
            "S_max"       : float('inf'), 
            "S_noise"     : 1, 
    }

    metrics = calculate_metrics.calculate_metrics_from_generator(
        network_pkl="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
        ref_path="data/refs/edm-1-imagenet-64x64.npz",
        max_batch_size=196,
        num_images=1000,
        sampler_kwargs=sampler_kwargs,
        # outdir="out",
    )

    print(metrics)


