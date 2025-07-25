import os 
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
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    logs_path = "data/runs.json"
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
        feature_dir = "data/features",
        template_dir = "data/templates_per_classid"
    )

    # Track duration of the run
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds() / 60
    
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
