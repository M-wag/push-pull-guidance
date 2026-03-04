"""Miscellaneous utility classes and functions."""

import os
import torch
from typing import Any
from torchvision.io import read_image, ImageReadMode

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
# Helper functions for loading in images from a path

def _load_images(path, device=None, dtype=None, for_torch=True, rescale=False):
    if path is None:
        return None
    elif os.path.isfile(path):
        templates = read_image(path, mode=ImageReadMode.RGB).unsqueeze(0)
    elif os.path.isdir(path):
        imgs = []
        for fname in sorted(os.listdir(path)): # iterate through each file in directory
            fpath = os.path.join(path, fname)
            if not os.path.isfile(fpath):
                continue
            imgs.append(read_image(fpath))
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
        templates = (templates - 128) / 127.5
    return templates

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
