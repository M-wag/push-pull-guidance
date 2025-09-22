import os 
import importlib.util
import csv
import re
import json
import torch
import calculate_metrics

from configs import gvf_sd as cfg 
from datetime import datetime, timezone
from torch_utils import distributed as dist
from dnnlib.util import EasyDictNested

#----------------------------------------------------------------------------
# Helper functions for importing saved data

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

def log_run_record(log_path: str, record: dict) -> None:
    with open(log_path, "a") as f:
        f.write(json.dumps(record, indent=4, default=str) + "\n")

def load_csv(path: str):
    with open(path, "r") as f:
        return [(int(col) for col in row) for row in csv.reader(f)] 

def load_features(metric, run_dir: str, template_dir: str):
    """ Loads feature vectors from a run and matches them to corresponding template features 
    based on (class_id, example_id) pairs from CSV files. """

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
# Class for runnig metrics for images

class ExperimentRunner:
    def __init__(self, paths: dict, max_batch_size: int = 128):

        self.paths  = EasyDictNested(paths)
        self.max_batch_size = max_batch_size

        if os.path.exists(self.paths.logs):
            self.run_id = get_next_run_id(self.paths.logs)
        else:
            self.run_id = 0

        config = load_vars_from_pyfile(self.paths.config)
        self.config = EasyDictNested(config)
        
    def run(self):
        # Calculate metrics and time duration
        start_time = datetime.now(timezone.utc)
        metrics = calculate_metrics.calculate_metrics_from_generator(
            network_pkl     = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-imagenet-64x64-cond-adm.pkl",
            max_batch_size  = self.max_batch_size,
            ref_path        = self.paths.refs,
            feature_dir     = self.paths.features,
            template_dir    = self.paths.templates,
            outdir          = self.paths.out,
            verbose         = False,
            subdirs         = True,
            generate_kwargs = self.config.generate_kwargs,
            sampler_kwargs  = self.config.sampler_kwargs,
            gradient_kwargs = self.config.gradient_kwargs,
            gvf_kwargs      = self.config.gvf_kwargs
        )
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds() / 60
        
        # Compute feature-dependent metrics
        self.add_feature_metrics(metrics)

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

    def add_feature_metrics(self, metrics):
         for metric in list(metrics.keys()) + ["clip"]:
            features_run, features_templates = load_features(metric, run_dir=self.paths.features, template_dir="data/features/examples")
            metrics[f"{metric}_csmean"] = torch.nn.functional.cosine_similarity(features_run, features_templates).mean().item()

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
    log_run_record(paths["logs"], run_record)
                
if __name__ == "__main__":
    main()
