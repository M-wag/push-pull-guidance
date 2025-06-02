import calculate_metrics

if __name__ == "__main__":
    metrics = calculate_metrics.calculate_metrics_from_generator(
        network_pkl="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
        ref_path="data/refs/edm-1-imagenet-64x64.npz",
        num_images=50
    )

    print(metrics)


