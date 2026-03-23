# PPG Image Gallery — Design

## Goal

A reusable library for generating images across many parameter combinations, saving them to disk, and building interactive HTML viewers. Replaces ad-hoc sweep scripts with a clean separation between **generation** and **viewing**.

## Architecture

```
sweeper/
├── __init__.py        # exports Gallery
├── gallery.py         # core library (generate + build_html)
├── viewer.py          # HTML generation
└── configs/           # YAML sweep definitions
    └── example.yaml
```

### Two phases, fully decoupled

**Phase 1: Generate** — `gallery.generate(config, build_fn)`

- Reads a YAML config that defines the parameter grid
- For each combination, calls `build_fn(params)` to configure the model, then runs inference
- Saves each output image as a PNG file in a structured directory
- Saves a `manifest.json` alongside the images recording every parameter combination and its output paths
- Skips images that already exist on disk (crash recovery)

**Phase 2: View** — `gallery.build_html(manifest_path, output_path)`

- Reads `manifest.json`
- Builds a static HTML page with two viewing modes:
  - **Grid mode**: pick two parameters for rows/columns, filter the rest
  - **Single mode**: show one image at a time, vary parameters with dropdowns and sliders
- Images are embedded as base64 in the HTML (self-contained, portable)
- Examples and prompts are displayed alongside each generated image

### YAML config format

```yaml
output_dir: sweeps/my_experiment

# These are shown alongside every generated image
examples:
  dataset: data/wild-ti2i
  n_entries: 3

# Fixed parameters (not swept)
fixed:
  num_inference_steps: 50
  guidance_scale: 7.5

# Parameters to sweep — each is a named axis
axes:
  noise_source:
    values: [random, ddim_inversion]

  projection:
    values: [gaussian, orthonormal]

  dim_out:
    values: [500, 1000, 2000, 4000, 8000]

  nu:
    linspace: [0.001, 1.0, 10]   # start, stop, num

  gate_type:
    values: [quadratic, heaviside]
```

Each axis defines either explicit `values` or `linspace` for numeric ranges. The full grid is the cartesian product of all axes.

### Output directory structure

```
sweeps/my_experiment/
├── manifest.json
└── images/
    ├── 0000_random_gaussian_500_0.001_quadratic/
    │   ├── img_0.png
    │   ├── img_1.png
    │   └── img_2.png
    ├── 0001_random_gaussian_500_0.001_heaviside/
    │   └── ...
    └── ...
```

Each subdirectory holds one image per example/prompt. The numeric prefix ensures stable ordering.

### manifest.json

```json
{
  "examples": ["data/wild-ti2i/data/img1.jpg", ...],
  "prompts": ["a cat on a beach", ...],
  "axes": ["noise_source", "projection", "dim_out", "nu", "gate_type"],
  "fixed": {"num_inference_steps": 50, "guidance_scale": 7.5},
  "entries": [
    {
      "params": {
        "noise_source": "random",
        "projection": "gaussian",
        "dim_out": 500,
        "nu": 0.001,
        "gate_type": "quadratic"
      },
      "images": ["images/0000_.../img_0.png", "images/0000_.../img_1.png", ...]
    },
    ...
  ]
}
```

### HTML viewer

Single self-contained HTML file with embedded JS. No server required.

**Grid mode:**
- Two dropdowns at top: "Rows" and "Columns" (populated from axis names)
- Remaining axes get filter dropdowns (pick one value each)
- Renders an image table: row labels, column labels, images in cells
- One tab per example image (or stacked vertically)

**Single mode:**
- One image displayed large
- Each axis gets a control: dropdown for categorical, slider for numeric
- Changing any control swaps the displayed image
- Example image shown alongside for comparison

Both modes show the prompt text beneath each image.

### gallery.py — API

```python
class Gallery:
    """Manages a sweep directory: generation + HTML building."""

    def __init__(self, config_path: str):
        """Load YAML config."""

    def generate(self, build_fn, run_fn):
        """
        Run the full sweep.

        build_fn(params: dict) — configure model for these params
        run_fn() -> list[np.ndarray] — run inference, return images
        """

    def build_html(self, output_path: str = None):
        """Build HTML viewer from manifest."""

    @staticmethod
    def load_manifest(manifest_path: str) -> dict:
        """Load an existing manifest (for building HTML from previous runs)."""
```

Usage in a script:

```python
gallery = Gallery("sweeper/configs/my_sweep.yaml")

def build(params):
    # configure dynamics.ppg based on params
    ...

def run():
    # run inference, return list of images
    ...

gallery.generate(build, run)
gallery.build_html("sweeps/my_experiment/viewer.html")
```

### Key design decisions

1. **Images on disk, not in memory** — allows crash recovery and avoids OOM for large sweeps
2. **Manifest is the contract** — generation writes it, viewer reads it. You can regenerate HTML without re-running inference
3. **build_fn / run_fn split** — gallery handles the loop and caching, caller handles model configuration and inference. This keeps the library model-agnostic
4. **Base64 embedding in HTML** — single portable file, no need for a file server. May get large for huge sweeps but practical for typical grids
5. **Two viewing modes** — grid is best for comparing two axes at a time, single+sliders is best for exploring high-dimensional spaces
