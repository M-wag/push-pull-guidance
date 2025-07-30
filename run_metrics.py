import torch
import os 
import csv
import json
from datetime import datetime, timezone
import calculate_metrics
from torch_utils import distributed as dist


# 50_000 images, template from custom image net
# gvf 
# sampler : 32
# metrics : L2-Norm


def get_next_run_id(log_path: str) -> int:
    """Get next available run ID from log file"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    last_id = 0
    with open(log_path, 'r') as f:
        for line in f:
            if stripped := line.strip():
                try:
                    record = json.loads(stripped)
                    last_id = max(last_id, record.get("run_id", 0))
                except json.JSONDecodeError:
                    continue
    return last_id + 1

def log_run_record(log_path: str, record: dict) -> None:
    """Append run record to log file"""
    with open(log_path, "a") as f:
        f.write(json.dumps(record, indent=4) + "\n")

def load_csv(path: str):
    with open(path, "r") as f:
        return [(int(col) for col in row) for row in csv.reader(f)] 

def load_features(metric, run_dir: str, template_dir: str):
    """ Loads feature vectors from a run and matches them to corresponding template features based on (class_id, example_id) pairs from CSV files. """

    # Load in features and metadata
    features_run = torch.load(os.path.join(run_dir, f"{metric}.pt"))
    features_template_all = torch.load(os.path.join(template_dir , f"{metric}.pt"))

    metadata_run = load_csv(os.path.join(run_dir, "features.csv"))
    metadata_templates = load_csv(os.path.join(template_dir, "features.csv"))

    # Create lookup dictionary for templates: (class_id, example_id) -> feature index
    template_id_to_index = {
            (class_id, example_id) : idx
            for idx, (class_id, example_id) in enumerate(metadata_templates)
    }

    # Find corresoding feature
    template_indices = [] 
    for class_id, example_id in metadata_run:
        try:
            template_indices.append(template_id_to_index[class_id, example_id])
        except:
            raise ValueError(
                f"No matching template found for (class_id={class_id}, example_id={example_id})"
            )

    # Select corresponding template features
    features_templates_matched = features_template_all[template_indices]

    assert features_templates_matched.shape == features_run.shape

    return features_run, features_templates_matched 

def cosine_similarity(x: torch.Tensor, y: torch.Tensor):
    """Cosine similarity between x (N, D) and y (N,D)"""
    x_norm = torch.nn.functional.normalize(x, dim=1)
    y_norm = torch.nn.functional.normalize(y, dim=1)
    return (x_norm * y_norm).sum(dim=1)
              

if __name__ == "__main__":
    logs_path = "data/runs.json"
    feature_dir = "data/features"
    template_dir = "data/templates_per_classid"

    num_images = 392
    network_pkl = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl"
    sampler_prms = {
            "num_steps"   : 16, 
            "sigma_min"   : 0.002  , 
            "sigma_max"   : 80, 
            "rho"         : 7, 
            "S_churn"     : 0.0,  
            "S_min"       : 0.0, 
            "S_max"       : float('inf'), 
            "S_noise"     : 1, 
    }
    gvf_prms = {}

    # Execute run and save features
    start_time = datetime.now(timezone.utc)
    metrics = calculate_metrics.calculate_metrics_from_generator(
        network_pkl=network_pkl,
        # ref_path="data/refs/edm-1-imagenet-64x64.npz",
        ref_path="data/refs/edm-2-imagenet-64x64.pkl",
        max_batch_size=128,
        num_images=num_images,
        sampler_kwargs=sampler_prms,
        # outdir="out",
        feature_dir = feature_dir,
        template_dir = template_dir,
        verbose=False,
    )

    # Track duration of the run
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds() / 60

    # Measure cosine similarity
    for metric in list(metrics.keys()):
        features_run, features_templates = load_features(metric, run_dir=feature_dir, template_dir="data/features_per_classid")
        metrics[f"{metric}_csmean"] = cosine_similarity(features_run, features_templates).mean().item()


    if dist.get_rank() == 0:
        # Get next run ID
        run_id = get_next_run_id(logs_path)
        
        # Prepare complete record
        run_record  = {
                "run_id"    : run_id,
                "datetime"  : start_time.isoformat(),
                "duration"  : duration,
                "num_images": num_images,
                "sampler "  : sampler_prms,
                "gvf"       : gvf_prms,
                "metrics"   : metrics,
        }

        # Save results
        log_run_record(logs_path, run_record)
        
        # Print summary
        print(f"Run {run_id} completed in {run_record['duration']:.2f} mins")
        for metric, value in metrics.items():
            print(f"{metric} : {value:.02f}")

