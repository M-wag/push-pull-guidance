import os 
import importlib.util
import re
import json
import torch
import tqdm
import calculate_metrics as cm 

from datetime import datetime, timezone
from torch_utils import distributed as dist
from dnnlib.util import EasyDictNested


#----------------------------------------------------------------------------
# Determine run id from logs file.

def get_next_run_id(log_path: str) -> int:
    """Get next available run ID from log file"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")
    
    last_id = 0
    with open(log_path, 'r') as f:
        for line in f:
            if stripped := line.strip():
                match = re.search(r'"run_id":\s*(\d+)', stripped)
                if match:
                    number = int(match.group(1))  # This will be "1"
                    last_id = max(last_id, number)
    return last_id + 1

#----------------------------------------------------------------------------
# Log the record of ran experiment.

def log_run_record(log_path: str, record: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(record, indent=4, default=str) + "\n")

#----------------------------------------------------------------------------
# Import variables ending with "kwargs" by name from a .py file

def load_vars_from_pyfile(filename, endswith="kwargs") -> dict[str, dict]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Config file not found: {filename}")
    
    # Get spec from file name 
    module_name = os.path.splitext(os.path.basename(filename))[0]
    spec = importlib.util.spec_from_file_location(module_name, filename)
    
    if spec is None:
        raise ValueError(f"Could not load spec from file: {filename}")
    
    module = importlib.util.module_from_spec(spec)

    # Execute module 
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise ValueError(f"Error executing module {filename}: {e}")

    # Store only the dictionaries in module
    dicts = {}
    for name in dir(module):
        if not name.endswith(endswith): # skip dunder methods
            continue 
        attr = getattr(module, name)
        dicts[name] = attr

    return dicts

#----------------------------------------------------------------------------
# Class for running different experiments involve image generation 
# and calculate of metrics

class ExperimentRunner:
    def __init__(self, 
                 paths: dict, 
                 max_batch_size: int = 128,
                 verbose : bool = False,
    ):

        self.paths  = EasyDictNested(paths)
        self.max_batch_size = max_batch_size
        self.verbose = verbose

        if os.path.exists(self.paths.logs):
            self.run_id = get_next_run_id(self.paths.logs)
        else:
            self.run_id = 0

        config = load_vars_from_pyfile(self.paths.config)
        self.config = EasyDictNested(config)
        
    def run(self):
        if not torch.distributed.is_initialized():
            dist.init()

        start_time = datetime.now(timezone.utc)

        image_iter = self.generate_images()
        metrics = self.calculate_metrics(image_iter)
        torch.distributed.barrier()

        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds() / 60

        if dist.get_rank() == 0:
            # Construct record
            run_record  = {
                    "run_id"    : self.run_id,
                    "datetime"  : start_time.isoformat(),
                    "duration"  : duration,
                    "generate"  : self.config.generate_kwargs,
                    "sampler"   : self.config.sampler_kwargs,
                    "gradient"  : self.config.gradient_kwargs,
                    "gvf"       : self.config.gvf_kwargs,
                    "metrics"   : metrics,
            }


            # Print metrics
            print(f"Run {self.run_id} completed in {run_record['duration']:.2f} mins")
            for metric, value in metrics.items():
                print(f"{metric} : {value:.02f}")

            return run_record


    def generate_images(self, seed=0):
        # Copy and pop num_images from generate_kwargs
        generate_kwargs = dict(self.config.generate_kwargs)
        num_images = generate_kwargs.pop("num_images")

        # Generate images
        seeds = range(seed, seed + num_images)
        image_iter = cm.generate.generate_images(
            net             = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
            seeds           = seeds,
            max_batch_size  = self.max_batch_size,
            verbose         = self.verbose,
            outdir          = self.paths.out,
            template_dir    = self.paths.templates,
            subdirs         = True,
            sampler_kwargs  = self.config.sampler_kwargs,
            gradient_kwargs = self.config.gradient_kwargs,
            gvf_args        = self.config.gvf_kwargs,
            **generate_kwargs,
        )

        return image_iter

    def calculate_metrics(self, image_iter, metrics = ['fid', 'fd_dinov2']):
        # Load reference stats
        if dist.get_rank() == 0:
            ref = cm.load_stats(self.paths.refs) # do this first in case it fails

        # Calculate statistics
        stats_iter = cm.calculate_stats_for_iterable(
            image_iter  = image_iter,
            metrics     = metrics,
            feature_dir = self.paths.features,
            verbose     = self.verbose
        )
        
        for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
            pass

        # Compute and return metrics
        results = {}
        if dist.get_rank() == 0:
            results = cm.calculate_metrics_from_stats(r.stats, ref, metrics, self.verbose)

        # Add custom metrics
        self.add_feature_metrics(results)
        
        return results

    def add_feature_metrics(self, metrics):
         for metric in list(metrics.keys()) + ["clip"]:
            features_run, features_templates = cm.load_features(metric, run_dir=self.paths.features, template_dir="data/features/examples")
            metrics[f"{metric}_csmean"] = torch.nn.functional.cosine_similarity(features_run, features_templates).mean().item()

#----------------------------------------------------------------------------

def main():
    run_id = get_next_run_id("data/runs.json")
    run_id = f"{run_id:04d}"
    paths = {
        "config"    : "configs/test.py",
        "logs"      : "data/runs.json",
        "features"  : f"data/features/run_{run_id}",
        "templates" : "data/images/examples",
        "refs"      : "data/refs/edm-1-imagnet-64x64.pkl",
        "out"       : "data/images/last"
    }

    runner = ExperimentRunner(paths)
    run_record = runner.run()
                
if __name__ == "__main__":
    main()
