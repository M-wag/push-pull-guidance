import calculate_metrics

if __name__ == "__main__":
    metrics = calculate_metrics.calculate_metrics_from_generator(
        network_pkl="path/to/model.pkl",
        ref_path="path/to/ref_stats.pkl",
        num_images=50
    )

    print(metrics)


