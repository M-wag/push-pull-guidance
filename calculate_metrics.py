# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate evaluation metrics (FID and FD_DINOv2)."""

import os
import csv
import shutil
import re
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset
import generate
from typing import Optional


#----------------------------------------------------------------------------
# Abstract base class for feature detectors.

class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x): # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# InceptionV3 feature detector.
# This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

class InceptionV3Detector(Detector):
    def __init__(self):
        super().__init__(feature_dim=2048)
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=False) as f:
            self.model = pickle.load(f)

    def __call__(self, x):
        return self.model.to(x.device)(x, return_features=True)

#----------------------------------------------------------------------------
# DINOv2 feature detector.
# Modeled after https://github.com/layer6ai-labs/dgm-eval

class DINOv2Detector(Detector):
    def __init__(self, resize_mode='torch'):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if self.resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif self.resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)

#----------------------------------------------------------------------------
# Metric specifications.

metric_specs = {
    'fid':          dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'fd_dinov2':    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector)),
}

#----------------------------------------------------------------------------
# Get feature detector for the given metric.

_detector_cache = dict()

def get_detector(metric, verbose=True):
    # Lookup from cache.
    if metric in _detector_cache:
        return _detector_cache[metric]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Construct detector.
    kwargs = metric_specs[metric].detector_kwargs
    if verbose:
        name = kwargs.class_name.split('.')[-1] if isinstance(kwargs.class_name, str) else kwargs.class_name.__name__
        dist.print0(f'Setting up {name}...')
    detector = dnnlib.util.construct_class_by_name(**kwargs)
    _detector_cache[metric] = detector

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    return detector

#----------------------------------------------------------------------------
# Load feature statistics from the given .pkl or .npz file.

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Save feature statistics to the given .pkl file.

def save_stats(stats, path, verbose=True):
    if verbose:
        print(f'Saving feature statistics to {path} ...')
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(stats, f)

#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                         # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics     = ['fid', 'fd_dinov2'], # Metrics to compute the statistics for.
    verbose     = True,                 # Enable status prints?
    dest_path   = None,                 # Where to save the statistics. None = do not save.
    feature_dir = None,                 # Where to save the feature. None = do not save.
    device      = torch.device('cuda'), # Which compute device to use.
):
    # Initialize.
    num_batches = len(image_iter)
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Convenience wrapper for torch.distributed.all_reduce().
    def all_reduce(x):
        x = x.clone()
        torch.distributed.all_reduce(x)
        return x

    # Return an iterable over the batches.
    class StatsIterable:
        def __len__(self):
            return num_batches

        def save_features(self, features_per_metric, feature_dir, batch_idx, images):
            rank = dist.get_rank()
            # Save features in parts for each metric
            for metric, features in features_per_metric.items():
                metric_dir = os.path.join(feature_dir, metric)
                os.makedirs(metric_dir, exist_ok=True) 
                file_path = f"features.rank{rank:01d}.part{batch_idx:04d}.pt"
                torch.save(features.cpu().detach(), os.path.join(metric_dir, file_path))

            # If classes and examples are passed also save CSV
            def is_one_hot(vec):
                vec = vec.to(torch.int)
                return vec.dim() == 1 and torch.all((vec == 0) | (vec == 1)) and vec.sum() == 1

            if isinstance(images, dict) and 'images' in images: # dict(images)
                classes = torch.argmax(images.labels, axis=1)
                examples = images.examples
            elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                labels = images[1]
                if labels is None:
                    classes = None
                    examples = None
                else:
                    if is_one_hot(labels):
                        classes = torch.argmax(labels, axis=1)
                    else:
                        classes = labels
                    examples = [None] * classes.shape[0]

            if classes is not None and examples is not None: 
                with open(os.path.join(feature_dir, f"features.rank{rank:01d}.part{batch_idx:04d}.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerows(zip(classes.tolist(), examples))

        def __iter__(self):
            state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]
            for s in state:
                s.cum_mu = torch.zeros([s.detector.feature_dim], dtype=torch.float64, device=device)
                s.cum_sigma = torch.zeros([s.detector.feature_dim, s.detector.feature_dim], dtype=torch.float64, device=device)
            cum_images = torch.zeros([], dtype=torch.int64, device=device)

            # Loop over batches.
            for batch_idx, images in enumerate(image_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    imgs = images.images
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    imgs = images[0]
                imgs = torch.as_tensor(imgs).to(device)

                # Accumulate statistics.
                features_per_metric = {}
                if imgs is not None:
                    for s in state:
                        features = s.detector(imgs).to(torch.float64)
                        s.cum_mu += features.sum(0)
                        s.cum_sigma += features.T @ features
                        features_per_metric[s.metric]= features
                    cum_images += imgs.shape[0]

                    # Save features of each image 
                    if feature_dir:
                        self.save_features(features_per_metric, feature_dir, batch_idx, images)

                # Output results.
                r = dnnlib.EasyDict(stats=None, images=images, batch_idx=batch_idx, num_batches=num_batches)
                r.num_images = int(all_reduce(cum_images).cpu())
                if batch_idx == num_batches - 1:
                    assert r.num_images >= 2
                    r.stats = dict(num_images=r.num_images)
                    for s in state:
                        mu = all_reduce(s.cum_mu) / r.num_images
                        sigma = (all_reduce(s.cum_sigma) - mu.ger(mu) * r.num_images) / (r.num_images - 1)
                        r.stats[s.metric] = dict(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
                    if dest_path is not None and dist.get_rank() == 0:
                        save_stats(stats=r.stats, path=dest_path, verbose=False)
                yield r

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_files(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 2,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    use_labels      = False,# Load to iterable with labels
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # List images.
    if verbose:
        dist.print0(f'Loading images from {image_path} ...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_images, random_seed=seed, use_labels=use_labels)
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)

#----------------------------------------------------------------------------
# Calculate metrics based on the given feature statistics.

def calculate_metrics_from_stats(
    stats,                          # Feature statistics of the generated images.
    ref,                            # Reference statistics of the dataset. Can be a path or URL.
    metrics = ['fid', 'fd_dinov2'], # List of metrics to compute.
    verbose = True,                 # Enable status prints?
):
    if isinstance(ref, str):
        ref = load_stats(ref, verbose=verbose)
    results = dict()
    for metric in metrics:
        if metric not in stats or metric not in ref:
            if verbose:
                print(f'No statistics computed for {metric} -- skipping.')
            continue
        if verbose:
            print(f'Calculating {metric}...')
        m = np.square(stats[metric]['mu'] - ref[metric]['mu']).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(stats[metric]['sigma'], ref[metric]['sigma']), disp=False)
        value = float(np.real(m + np.trace(stats[metric]['sigma'] + ref[metric]['sigma'] - s * 2)))
        results[metric] = value
        if verbose:
            print(f'{metric} = {value:g}')
    return results

#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics

#---------------------------------------------------------------------------
# Merge the seperate feature files

def merge_metric_feature_directories(base_dir, delete_dirs=True):
    """
    Merge feature part files organized by metric directories into single .pt files
    
    Args:
        base_dir: Directory containing metric subdirectories
        delete_dirs: Whether to delete original subdirectories after merging
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")
    
    # Process each metric subdirectory
    metrics = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    for metric_name in metrics:
        metric_dir = os.path.join(base_dir, metric_name)
        output_file = os.path.join(base_dir, f"{metric_name}.pt")
        
        # Collect and sort feature files
        feature_files = []
        for fname in os.listdir(metric_dir):
            if fname.endswith(".pt") and "features" in fname:
                # Extract ordering information
                parts = re.findall(r"rank(\d+).part(\d+)", fname)
                if parts:
                    rank, part = map(int, parts[0])
                    feature_files.append((rank, part, fname))
        
        if not feature_files:
            print(f"No feature files found in {metric_dir}")
            continue
        
        # Sort by rank then part number
        feature_files.sort(key=lambda x: (x[0], x[1]))
        
        # Load and concatenate features
        all_features = []
        for _, _, fname in feature_files:
            file_path = os.path.join(metric_dir, fname)
            features = torch.load(file_path)
            all_features.append(features)
        
        # Handle variable sizes
        try:
            combined = torch.cat(all_features, dim=0)
        except RuntimeError:
            # Save as list if sizes don't match
            combined = all_features
        
        torch.save(combined, output_file)
        print(f"Merged {metric_name} features to {output_file} "
              f"({len(all_features)} files merged)")

        # Cleanup
        if delete_dirs:
            shutil.rmtree(metric_dir)

def merge_feature_csvs(csv_dir, output_path="features.csv", delete_parts=True):
    """Merge CSV files from feature extraction"""
    csv_files = []
    for fname in os.listdir(csv_dir):
        if fname.endswith(".csv") and "features" in fname:
            parts = re.findall(r"rank(\d+).part(\d+)", fname)
            if parts:
                rank, part = map(int, parts[0])
                csv_files.append((rank, part, fname))
    
    if not csv_files:
        print(f"No csv files found in {csv_dir}")
        return
    
    csv_files.sort(key=lambda x: (x[0], x[1]))
    
    with open(os.path.join(csv_dir, output_path), "w") as outfile:
        # Append content
        for _, _, fname in csv_files:
            file_path = os.path.join(csv_dir, fname)
            with open(file_path) as infile:
                # infile.readline()  # Skip header
                outfile.write(infile.read())

    # Cleanup part files
    if delete_parts:
        for _, _, fpath in csv_files:
            os.remove(os.path.join(csv_dir, fpath))

    print(f"Merged csvs to {output_path} "
          f"({len(csv_files)} files merged)")

#----------------------------------------------------------------------------
# Calculate metrics based on the given feature statistics.

def calculate_metrics_from_directory(
    image_path: str,
    ref_path: str,
    metrics: list[str] = ['fid', 'fd_dinov2'],
    num_images: int = 50000,
    seed: int = 0,
    max_batch_size: int = 64,
    num_workers: int = 2,
    verbose: bool = True,
) -> dict[str, float]:
    """Calculate metrics for images in a directory/ZIP file."""

    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    torch.distributed.barrier()

#----------------------------------------------------------------------------
# Calculate metrics for a generative model

def calculate_metrics_from_generator(
    network_pkl:    str,
    ref_path:       str,
    metrics:        list[str] = ['fid', 'fd_dinov2'],
    num_images:     int = 50000,
    seed:           int = 0,
    max_batch_size: int = 32,
    verbose:        bool = True,
    sampler_kwargs: dict = None,
    cfg_gvf         = None,                 # Config for Guidance Vector Field
    outdir          = None,                 # Where to save the output images. None = do not save.
    feature_dir     = None,                 # Where to save the features of images. None = do not save.
    subdirs         = False,                # Create subdirectory for every 1000 seeds?
    template_dir    = None                  # Where templates are stored
) -> dict[str, float]:
    """Calculate metrics for a generative model."""
    dist.init()
    
    # Load reference stats
    if dist.get_rank() == 0:
        ref = load_stats(ref_path) # do this first in case it fails
    
    # Generate images
    seeds = range(seed, seed + num_images)
    image_iter = generate.generate_images(
        net=network_pkl,
        gvf_args=cfg_gvf,
        seeds=seeds,
        max_batch_size=max_batch_size,
        outdir=outdir,
        subdirs=subdirs,
        sampler_kwargs=sampler_kwargs,
        template_dir=template_dir,
    )
    
    # Calculate statistics
    stats_iter = calculate_stats_for_iterable(
        image_iter=image_iter,
        metrics=metrics,
        feature_dir=feature_dir,
        verbose=verbose
    )
    
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    
    # Merge files in case features were saved
    if feature_dir and dist.get_rank() == 0:
        merge_metric_feature_directories(feature_dir)
        merge_feature_csvs(feature_dir)


    # Compute and return metrics
    results = {}
    if dist.get_rank() == 0:
        results = calculate_metrics_from_stats(r.stats, ref, metrics, verbose)
    
    torch.distributed.barrier()
    return results

#----------------------------------------------------------------------------
# Calculate reference statistics for a dataset

def generate_reference_stats(
    image_path: str,
    dest_path: str,
    metrics: list[str] = ['fid', 'fd_dinov2'],
    max_batch_size: int = 64,
    num_workers: int = 2,
    verbose: bool = True,
    feature_dir = None,
    use_labels = False, 
) -> Optional[dict]:
    """Generate reference statistics for a dataset."""
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()
    
    # Calculate and save statistics
    stats_iter = calculate_stats_for_files(
        image_path=image_path,
        metrics=metrics,
        max_batch_size=max_batch_size,
        num_workers=num_workers,
        verbose=verbose,
        dest_path=dest_path,
        feature_dir=feature_dir,
        use_labels=use_labels,
    )
    
    # Process batches
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    
    # Return stats on rank 0
    if dist.get_rank() == 0:
        return r.stats
    return None

