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
import transformers

from abc import ABC, abstractmethod
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset
from typing import Any, Optional
from util import EasyDict


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
# CLIP Model 

class ClipDetector(Detector):
    def __init__(self):
        super().__init__(feature_dim=768)
        self.model = transformers.CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = transformers.AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        inputs = self.processor(images=x, return_tensors="pt").to(x.device)
        with torch.no_grad():
            return self.model.to(x.device).get_image_features(**inputs)

#----------------------------------------------------------------------------
# Metric specifications.

metric_specs = {
    'inception': InceptionV3Detector,
    'dinov2':    DINOv2Detector,
    'clip':      ClipDetector,
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
    if verbose:
        dist.print0(f'Setting up {metric_specs[metric].__name__}...')
    detector = metric_specs[metric]()
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
            return {'inception': dict(np.load(f))}
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

#---------------------------------------------------------------------------
# Merge the seperate .pt feature files 

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

#---------------------------------------------------------------------------
# Merge the seperate .csv feature files 

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
# Load feature values from the given .pkl file.

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
# Load feature example and id from csv file

def load_csv(path: str):
    with open(path, "r") as f:
        return [(int(col) for col in row) for row in csv.reader(f)] 

#----------------------------------------------------------------------------
# Module-level helpers.

def _all_reduce(x):
    """Clone and torch.distributed.all_reduce; no-op if distributed not init."""
    x = x.clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(x)
    return x


def _is_one_hot(vec):
    """Check if `vec` is a 1-D one-hot tensor."""
    vec = vec.to(torch.int)
    return vec.dim() == 1 and torch.all((vec == 0) | (vec == 1)) and vec.sum() == 1


def _extract_images_and_meta(batch):
    """Unpack a batch from an image iterable.

    Supports two shapes:
      - EasyDict state (from generate.generate_images): has .images tensor,
        optionally .class_id / .example_id from MetadataIterable.
      - (imgs, labels) tuple/list (from torch DataLoader).

    Returns (imgs_tensor, class_id_list_or_None, example_id_list_or_None).
    """
    if isinstance(batch, dict) and 'images' in batch:
        imgs = batch['images']
        class_id = batch.get('class_id')
        example_id = batch.get('example_id')
        if class_id is None and 'labels' in batch:
            class_id = torch.argmax(batch['labels'], axis=1)
            example_id = batch.get('example_idx')
        if class_id is not None and isinstance(class_id, torch.Tensor):
            class_id = class_id.tolist()
        if example_id is None and class_id is not None:
            example_id = [None] * len(class_id)
        return imgs, class_id, example_id

    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        imgs, labels = batch
        if labels is None:
            return imgs, None, None
        if _is_one_hot(labels):
            class_id = torch.argmax(labels, axis=1).tolist()
        else:
            class_id = labels.tolist() if isinstance(labels, torch.Tensor) else list(labels)
        return imgs, class_id, [None] * len(class_id)

    raise TypeError(f'Unsupported batch type: {type(batch)}')


def _save_features(features_per_metric, feature_dir, batch_idx, class_id, example_id):
    """Save per-rank feature parts (.pt) and metadata (.csv) for one batch."""
    rank = dist.get_rank()
    for metric, features in features_per_metric.items():
        metric_dir = os.path.join(feature_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        file_path = f"features.rank{rank:01d}.part{batch_idx:04d}.pt"
        torch.save(features.cpu().detach(), os.path.join(metric_dir, file_path))

    if class_id is not None and example_id is not None:
        with open(os.path.join(feature_dir, f"features.rank{rank:01d}.part{batch_idx:04d}.csv"), "w") as f:
            csv.writer(f).writerows(zip(class_id, example_id))


#----------------------------------------------------------------------------
# Compute detector features for every batch in an image iterable.
# Yields EasyDict(features, images, batch_idx, num_batches, num_images_in_batch).

def compute_features_for_iterable(
    image_iter,                                         # Iterable of image batches.
    metrics         = ('inception', 'dinov2'),             # Metric names (keys of metric_specs).
    feature_dir     = None,                             # If set, save per-rank feature parts here.
    verbose         = True,                             # Enable status prints.
    device          = torch.device('cuda'),             # Compute device.
):
    num_batches = len(image_iter)
    detectors = {m: get_detector(m, verbose=verbose) for m in metrics}
    if verbose:
        dist.print0('Computing detector features...')

    for batch_idx, batch in enumerate(image_iter):
        imgs, class_id, example_id = _extract_images_and_meta(batch)
        imgs = torch.as_tensor(imgs).to(device)

        features = {m: detectors[m](imgs).to(torch.float64) for m in metrics}

        if feature_dir:
            _save_features(features, feature_dir, batch_idx, class_id, example_id)

        yield EasyDict(
            features            = features,
            images              = batch,
            batch_idx           = batch_idx,
            num_batches         = num_batches,
            num_images_in_batch = imgs.shape[0],
        )

    if feature_dir and dist.get_rank() == 0:
        merge_metric_feature_directories(feature_dir)
        merge_feature_csvs(feature_dir)


#----------------------------------------------------------------------------
# Generic reducer. Consumes any iterable of batches and accumulates state
# via `per_batch_fn(state, batch) -> state`. On the last batch, calls
# `final_fn(state, batch) -> result` and attaches the result onto the yielded
# batch as `.result`. Earlier batches yield with `.result = None`.

def reduce_iterable(iterable, per_batch_fn, final_fn, init_state=None):
    state = init_state
    for batch in iterable:
        state = per_batch_fn(state, batch)
        batch.result = final_fn(state, batch) if batch.batch_idx == batch.num_batches - 1 else None
        yield batch


#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                             # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics         = ['inception', 'dinov2'], # Metrics to compute the statistics for.
    verbose         = True,                 # Enable status prints?
    dest_path       = None,                 # Where to save the statistics. None = do not save.
    feature_dir     = None,                 # Where to save the feature. None = do not save.
    device          = torch.device('cuda'), # Which compute device to use.
):
    num_batches  = len(image_iter)
    feature_dims = {m: get_detector(m, verbose=verbose).feature_dim for m in metrics}

    accumulator = EasyDict(
        per_metric = {m: EasyDict(
            cum_mu    = torch.zeros([feature_dims[m]], dtype=torch.float64, device=device),
            cum_sigma = torch.zeros([feature_dims[m], feature_dims[m]], dtype=torch.float64, device=device),
        ) for m in metrics},
        total_images = torch.zeros([], dtype=torch.int64, device=device),
    )

    def per_batch(acc, batch):
        for m in metrics:
            acc.per_metric[m].cum_mu    += batch.features[m].sum(0)
            acc.per_metric[m].cum_sigma += batch.features[m].T @ batch.features[m]
        acc.total_images += batch.num_images_in_batch
        return acc

    def final_fn(acc, batch):
        n = int(_all_reduce(acc.total_images).cpu())
        assert n >= 2
        stats = dict(num_images=n)
        for m in metrics:
            mu    = _all_reduce(acc.per_metric[m].cum_mu) / n
            sigma = (_all_reduce(acc.per_metric[m].cum_sigma) - mu.ger(mu) * n) / (n - 1)
            stats[m] = dict(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
        if dest_path is not None and dist.get_rank() == 0:
            save_stats(stats=stats, path=dest_path, verbose=False)
        return stats

    feature_iter = compute_features_for_iterable(
        image_iter, metrics=metrics, feature_dir=feature_dir, verbose=verbose, device=device,
    )

    class StatsIterable:
        def __len__(self):
            return num_batches
        def __iter__(self):
            for batch in reduce_iterable(feature_iter, per_batch, final_fn, init_state=accumulator):
                yield EasyDict(
                    stats       = batch.result,
                    images      = batch.images,
                    batch_idx   = batch.batch_idx,
                    num_batches = batch.num_batches,
                    num_images  = int(_all_reduce(accumulator.total_images).cpu()),
                )

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# EasyDict(stats, images, batch_idx, num_batches)

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
    metrics = ['inception', 'dinov2'], # List of metrics to compute.
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
# Transform classes — one per metric type. Each operates on pre-computed
# feature batches and is unaware of detectors or feature names.

class Transform(ABC):
    @abstractmethod
    def init(self, feature_names: list[str], device) -> Any:
        """Allocate and return a fresh accumulator."""

    @abstractmethod
    def update(self, acc, batch) -> Any:
        """Accumulate one feature batch. batch.features has all features."""

    @abstractmethod
    def finalize(self, acc) -> dict[str, float]:
        """All-reduce and return {feature_name: value}."""


class FDTransform(Transform):
    def __init__(self, ref_stats):
        self.ref_stats = ref_stats

    def init(self, feature_names, device):
        return {f: EasyDict(
            cum_mu    = torch.zeros([get_detector(f).feature_dim], dtype=torch.float64, device=device),
            cum_sigma = torch.zeros([get_detector(f).feature_dim] * 2, dtype=torch.float64, device=device),
            total     = torch.zeros([], dtype=torch.int64, device=device),
            name      = f,
        ) for f in feature_names}

    def update(self, acc, batch):
        for f, s in acc.items():
            feats = batch.features[f]
            s.cum_mu    += feats.sum(0)
            s.cum_sigma += feats.T @ feats
            s.total     += batch.num_images_in_batch
        return acc

    def finalize(self, acc) -> dict[str, float]:
        stats = {}
        n = int(_all_reduce(next(iter(acc.values())).total).cpu())
        for f, s in acc.items():
            mu    = _all_reduce(s.cum_mu) / n
            sigma = (_all_reduce(s.cum_sigma) - mu.ger(mu) * n) / (n - 1)
            stats[f] = dict(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
        stats['num_images'] = n
        return calculate_metrics_from_stats(stats, self.ref_stats, list(acc.keys()))


class CSTransform(Transform):
    def __init__(self, example_features_dir):
        self.dir = example_features_dir

    def init(self, feature_names, device):
        self.device = device
        self.example_features = {
            f: np.load(os.path.join(self.dir, f'{f}.npy'), mmap_mode='r')
            for f in feature_names
        }
        self.index = {
            (cls, eid): row
            for row, (cls, eid) in enumerate(load_csv(os.path.join(self.dir, 'index.csv')))
        }
        return {f: EasyDict(
            cum_sim = torch.zeros([], dtype=torch.float64, device=device),
            total   = torch.zeros([], dtype=torch.int64,   device=device),
        ) for f in feature_names}

    def update(self, acc, batch):
        class_ids   = batch.images.class_id
        example_ids = batch.images.example_id
        rows = [self.index[int(c), int(e)] for c, e in zip(class_ids, example_ids)]
        for f, s in acc.items():
            gen   = batch.features[f].to(torch.float32)
            ref   = torch.from_numpy(np.ascontiguousarray(self.example_features[f][rows]))
            ref   = ref.to(device=self.device, dtype=torch.float32)
            s.cum_sim += torch.nn.functional.cosine_similarity(gen, ref).sum().to(torch.float64)
            s.total   += batch.num_images_in_batch
        return acc

    def finalize(self, acc) -> dict[str, float]:
        results = {}
        for f, s in acc.items():
            n = int(_all_reduce(s.total).cpu())
            results[f] = float(_all_reduce(s.cum_sim).cpu()) / n
        return results


TRANSFORMS = {
    'fd': FDTransform,
    'cs': CSTransform,
}

#----------------------------------------------------------------------------
# Calculate metrics in a single pass over `image_iter`.
# Returns a flat dict of float results keyed as `{transform}_{feature}`.

def calculate_metrics_from_iterable(
    image_iter,
    ref_stats,                                           # Path or already-loaded {metric: {mu, sigma}}.
    metrics              = {'fd': ['inception', 'dinov2']},  # {transform: [feature, ...]}
    example_features_dir = None,                         # Required when 'cs' is in metrics.
    verbose              = True,
    device               = torch.device('cuda'),
) -> dict[str, float]:

    if isinstance(ref_stats, str):
        ref_stats = load_stats(ref_stats, verbose=verbose)

    transform_kwargs = {
        'fd': dict(ref_stats=ref_stats),
        'cs': dict(example_features_dir=example_features_dir),
    }
    transforms = {t: TRANSFORMS[t](**transform_kwargs[t]) for t in metrics}

    all_features = list(dict.fromkeys(f for fs in metrics.values() for f in fs))
    accs = {t: transforms[t].init(feature_names, device)
            for t, feature_names in metrics.items()}

    feature_iter = compute_features_for_iterable(
        image_iter, metrics=all_features, verbose=verbose, device=device,
    )

    for batch in tqdm.tqdm(feature_iter, unit='batch', total=len(image_iter),
                           disable=(dist.get_rank() != 0)):
        for t in transforms:
            accs[t] = transforms[t].update(accs[t], batch)

    results = {}
    for t in transforms:
        results.update({f'{t}_{k}': v for k, v in transforms[t].finalize(accs[t]).items()})
    return results


#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics


#----------------------------------------------------------------------------
# Calculate reference statistics for a dataset

def generate_reference_stats(
    image_path: str,
    dest_path: str,
    metrics: list[str] = ['inception', 'dinov2'],
    max_batch_size: int = 64,
    num_workers: int = 2,
    verbose: bool = True,
    feature_dir = None, use_labels = False, 
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



#----------------------------------------------------------------------------
# Command-line interface for calculating metrics

@click.group()
def cmdline():
 """Calculate evaluation metrics (FID and FD_DINOv2)."""

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='inception,dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc(ref_path, metrics, **opts):
    """Calculate metrics for a given set of images."""
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

if __name__ == "__main__":
    cmdline()
