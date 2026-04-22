"""Miscellaneous utility classes and functions."""

import glob
import hashlib
import html
import io
import os
import re
import requests
import tempfile
import urllib
import urllib.parse
import uuid
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, BinaryIO, Optional
from PIL import Image

#----------------------------------------------------------------------------
# Convenience class that behaves like a dict but allows access with the attribute syntax. x = d.key

class EasyDict(dict):

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.

def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(value, shape=shape, dtype=dtype, device=device, memory_format=memory_format)

#----------------------------------------------------------------------------
# Helper functions for loading in images from a path

def _read_image_pil(path):
    """Read image as (C, H, W) uint8 tensor using PIL."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)  # (H, W, 3) uint8
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

def _load_images(path, device=None, dtype=None, for_torch=True, rescale=False):
    if path is None:
        return None
    elif os.path.isfile(path):
        templates = _read_image_pil(path).unsqueeze(0)
    elif os.path.isdir(path):
        imgs = []
        for fname in sorted(os.listdir(path)): # iterate through each file in directory
            fpath = os.path.join(path, fname)
            if not os.path.isfile(fpath):
                continue
            imgs.append(_read_image_pil(fpath))
        if not imgs:
            return None
        templates = torch.stack(imgs)
    else:
        raise ValueError(
            f"Template path must be an existing file, directory, or None; "
            f"got {path!r} (type {type(path).__name__})"
    )

    if device:
        templates = templates.to(device=device)
    if dtype:
        templates = templates.to(dtype=dtype)
    if rescale:
        templates = templates.to(torch.float32) / 127.5 - 1 # MAKE SURE THIS IS IN FLOAT, WILL FAIL SILENTLY ON UINT8
    return templates

#----------------------------------------------------------------------------
# Logger: records per-step score diagnostics during denoising

class Logger:

    def __init__(self):
        self._step_batch = []       # list of {key: float} — batch-averaged per step
        self._step_samples = []     # list of {key: tensor(N)} — per-sample per step

    # ------------------------------------------------------------------
    # Public API

    def record_scores(self, s_comb, s_model, s_guide):
        """Compute score diagnostics for one denoising step and store them."""
        batch, per_sample = self._compute(s_comb, s_model, s_guide)
        self._step_batch.append(batch)
        self._step_samples.append(per_sample)

    def reset(self):
        self._step_batch = []
        self._step_samples = []

    def get_batch_logs(self):
        """Return list of per-step dicts with batch-averaged scalars."""
        return list(self._step_batch)

    def get_per_image_logs(self):
        """Return list[list[dict]]: outer = image index, inner = step."""
        if not self._step_samples:
            return []
        n = len(self._step_samples[0][next(iter(self._step_samples[0]))])
        return [
            [{k: float(v[i]) for k, v in step.items()} for step in self._step_samples]
            for i in range(n)
        ]

    def plot(self):
        """Return a matplotlib Figure with time-series plots of batch-averaged diagnostics."""
        logs = self._step_batch
        if not logs:
            return None

        steps = range(len(logs))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        for key, label in [("norm_comb_mean", "combined"), ("norm_model_mean", "model"), ("norm_guide_mean", "guide")]:
            axes[0].plot(steps, [d[key] for d in logs], label=label)
        axes[0].set_title("Score Norms")
        axes[0].set_xlabel("Step")
        axes[0].legend(fontsize=8)

        for key, label in [("cos_model_guide_mean", "model/guide"), ("cos_model_comb_mean", "model/comb"), ("cos_guide_comb_mean", "guide/comb")]:
            axes[1].plot(steps, [d[key] for d in logs], label=label)
        axes[1].set_title("Cosine Similarities")
        axes[1].set_xlabel("Step")
        axes[1].set_ylim(-1, 1)
        axes[1].legend(fontsize=8)

        for key, label in [("ratio_comb_model_mean", "‖comb‖/‖model‖"), ("ratio_guide_model_mean", "‖guide‖/‖model‖")]:
            axes[2].plot(steps, [d[key] for d in logs], label=label)
        axes[2].set_title("Score Ratios")
        axes[2].set_xlabel("Step")
        axes[2].legend(fontsize=8)

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _flatten(x):
        return x.view(x.shape[0], -1)

    @staticmethod
    def _norm(x):
        return Logger._flatten(x).norm(dim=1)  # (N,)

    @staticmethod
    def _cosine(a, b, eps=1e-8):
        af, bf = Logger._flatten(a), Logger._flatten(b)
        dot = (af * bf).sum(dim=1)
        return dot / (af.norm(dim=1) * bf.norm(dim=1) + eps)  # (N,)

    def _compute(self, s_comb, s_model, s_guide):
        norm_comb  = self._norm(s_comb)
        norm_model = self._norm(s_model)
        norm_guide = self._norm(s_guide)

        cos_model_guide = self._cosine(s_model, s_guide)
        cos_model_comb  = self._cosine(s_model, s_comb)
        cos_guide_comb  = self._cosine(s_guide, s_comb)

        ratio_comb_model  = norm_comb  / (norm_model + 1e-8)
        ratio_guide_model = norm_guide / (norm_model + 1e-8)

        per_sample = {
            "norm_comb":         norm_comb,
            "norm_model":        norm_model,
            "norm_guide":        norm_guide,
            "cos_model_guide":   cos_model_guide,
            "cos_model_comb":    cos_model_comb,
            "cos_guide_comb":    cos_guide_comb,
            "ratio_comb_model":  ratio_comb_model,
            "ratio_guide_model": ratio_guide_model,
        }
        batch = {
            "norm_comb_mean":          float(norm_comb.mean()),
            "norm_model_mean":         float(norm_model.mean()),
            "norm_guide_mean":         float(norm_guide.mean()),
            "norm_comb_std":           float(norm_comb.std()),
            "norm_model_std":          float(norm_model.std()),
            "norm_guide_std":          float(norm_guide.std()),
            "cos_model_guide_mean":    float(cos_model_guide.mean()),
            "cos_model_comb_mean":     float(cos_model_comb.mean()),
            "cos_guide_comb_mean":     float(cos_guide_comb.mean()),
            "ratio_comb_model_mean":   float(ratio_comb_model.mean()),
            "ratio_guide_model_mean":  float(ratio_guide_model.mean()),
        }
        return batch, per_sample


#----------------------------------------------------------------------------

def edm_sigmas(num_steps=32, sigma_min=0.002, sigma_max=80.0, rho=7):
    """Return the EDM sigma schedule as a (num_steps+1,) float64 tensor (last entry is 0)."""
    step_indices = torch.arange(num_steps, dtype=torch.float64)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([t_steps, t_steps.new_zeros(1)])

#----------------------------------------------------------------------------

def load_images(batch_template_info, device=None, dtype=None, for_torch=True, rescale=False):
    """
    batch_template_info: list of either paths, or list of filenames/indices to load from `template_dir`
    template_dir: if `batch_template_info` contains filenames or indices
    """

    batch_templates = []
    for entry in batch_template_info:
        result = _load_images(entry, device, dtype, for_torch, rescale)
        if result is not None:
            batch_templates.append(result)

    if not batch_templates:
        return None
    if for_torch:
        batch_templates = torch.concat(batch_templates)

    return batch_templates

#----------------------------------------------------------------------------

_cache_dir = None

def set_cache_dir(path: str) -> None:
    global _cache_dir
    _cache_dir = path

def make_cache_dir_path(*paths: str) -> str:
    if _cache_dir is not None:
        return os.path.join(_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

def _is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    if not isinstance(obj, str) or '://' not in obj:
        return False
    if allow_file_urls and obj.startswith('file://'):
        return True
    try:
        res = urllib.parse.urlparse(obj)
        if not res.scheme or not res.netloc or '.' not in res.netloc:
            return False
        res = urllib.parse.urlparse(urllib.parse.urljoin(obj, '/'))
        if not res.scheme or not res.netloc or '.' not in res.netloc:
            return False
    except:
        return False
    return True

def open_url(url: str, cache_dir: Optional[str] = None, num_attempts: int = 10,
             verbose: bool = True, return_filename: bool = False, cache: bool = True) -> BinaryIO:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and not cache)

    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, 'rb')  # type: ignore

    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, 'rb')  # type: ignore

    assert _is_url(url)

    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode('utf-8')).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + '_*'))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, 'rb')  # type: ignore

    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print('Downloading %s ...' % url, end='', flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError('No data received')
                    if len(res.content) < 8192:
                        content_str = res.content.decode('utf-8')
                        if 'download_warning' in res.headers.get('Set-Cookie', ''):
                            links = [html.unescape(link) for link in content_str.split('"') if 'export=download' in link]
                            if len(links) == 1:
                                url = urllib.parse.urljoin(url, links[0])
                                raise IOError('Google Drive virus checker nag')
                        if 'Google Drive - Quota exceeded' in content_str:
                            raise IOError('Google Drive download quota exceeded -- please try again later')
                    match = re.search(r'filename="([^"]*)"', res.headers.get('Content-Disposition', ''))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(' done')
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(' failed')
                    raise
                if verbose:
                    print('.', end='', flush=True)

    assert url_data is not None
    if cache:
        assert url_name is not None
        safe_name = re.sub(r'[^0-9a-zA-Z-._]', '_', url_name)
        safe_name = safe_name[:min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + '_' + safe_name)
        temp_file = os.path.join(cache_dir, 'tmp_' + uuid.uuid4().hex + '_' + url_md5 + '_' + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, 'wb') as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)
        if return_filename:
            return cache_file  # type: ignore

    assert not return_filename
    return io.BytesIO(url_data)
