
#   Push-Pull Guidance: Image Editing by Pushing-Forward the gradients of Latent Space
- Mathematical diagram

<p align="center">
  <video src="demo/fish.mp4" width="480" autoplay loop muted playsinline></video>
</p>

##  Official Pytorch Implemetation of Push-Pull Guidance

*(preprint coming soon)* 



Given an input image, Push-Pull Guidance allows you to generated similar images, by using off the-shelf denoising diffusuin models. The method requires no re-training or finetuning and only requires a user to provide an image and indicate how strongly they want generated image to match the input. To be more precise, diffusion models operate by evolving a stochastic differential equation (SDE). We add an additional term to the SDE which guides the generation process towards a target image. 

##  Exploring Different Parameters with Sweep Viewer
We visualize our results using `sweep.py` which calls a config.

```.bash
python sweep.py demo/config.yaml
```

The workload can be distributed across multiple GPUs by running the sweep command using `torchrun`:
```.bash
torchrun --nproc-per-node=4 sweep.py demo/config.yaml
```

### Defining a sweep

A sweep is declared by a YAML config.
 Any field written as a `values:` list (or a `linspace:`) is promoted to a **sweep axis**; the sweep then runs the full Cartesian product of all axes and renders one cell per combination. Scalar fields are held fixed across the whole grid. For more details on configurations look at `sweeper/schema.py`. Below we show a simplified config for doing guidance in pixel space.

```yaml
title: "EDM ImageNet 64x64 — Pixel Guidance Sweep"
output_dir: demo/sweeps/pushpull

model:
  type: edm                      # fixed: same model for every cell

examples:
  samples_yaml: data/imgnet64/my_samples.yaml
  n_entries: 10                  # images to edit
  n_seeds: 4                     # seeds per image

solver:                          # settings for EDM solver
  num_steps: 32
  stochastic: true
  second_order: true

ppg:                             # parameters for push-pull gradient
  mean_scale: ve
  normalize_variance:
    values: [none, decomposed]         # axis (2 values)
  gate:
    type: hill
    nu:
      values: [25.45, 13.26, 10.52, 6.46, 5.0, 2.9, 2.17, 1.61, 0.84]   # axis (9 values)
    n:
      values: [2, .inf]             # axis (2 value)

```


The config above yields a total of `2 x 9 x 2 = 36` cells.
### Browsing the results

Each run writes its images, a `manifest.json`, and a **HTML
viewer** into the designated output directory. Open `viewer.html` to explore the grid interactively. It offers two modes:

- **Grid mode** -- pick any two axes for the rows and columns and filter the
  rest, to compare a 2-D slice of the parameter space at a glance.
- **Single mode** -- focus on one cell with dropdowns/sliders for every axis,
  plus toggleable panels for the input example, baseline, generated output,
  per-step denoising timeline, and diagnostic plots.

![Sweep viewer screenshot](demo/viewer.jpeg)

##  Calculating Metrics for Different Parameters Settings

The viewer is for qualitative inspection; to evaluate a sweep quantitatively,
`metrics_sweep.py` runs the same parameter grid but, instead of building an HTML
viewer, computes a set of metrics for every cell and writes the results to a CSV.

```.bash
python metrics_sweep.py demo/config.yaml                          # single GPU
```

As with the sweep viewer, the workload distributes across multiple GPUs with
`torchrun` — metrics are accumulated across ranks, so the result is identical to
a single-GPU run:

```.bash
torchrun --nproc-per-node=4 metrics_sweep.py demo/config.yaml     # multi-GPU
```

### Configuring metrics

A metrics config is an ordinary sweep config with a few extra top-level fields
that describe what to measure and where to write it:

```yaml
# ... the same model / examples / ppg / maps blocks as a sweep config ...

ref_stats: data/refs/edm-1-imagnet-64x64.pkl   # reference dataset statistics (for FD)
example_features_dir: data/imgnet64/features   # cached features of the source images
metrics:
  cs: [clip, dinov2, pixel]                    # content similarity to the source
  pr: [dinov2]                                 # precision / recall
output_csv: results/pushpull/metrics.csv       # where the per-cell table is written
output_dir: results/pushpull
n_sample_images: 10                            # sample images to save per cell
```

The `metrics` field maps a **metric type** to the list of **feature extractors**
it should be computed over:

| Type | Meaning | Extractors |
|------|---------|------------|
| `fd` | Fréchet Distance to the reference dataset (à la FID / FD-DINOv2) — overall sample quality | `inception`, `dinov2` |
| `cs` | Mean cosine similarity between each edit and its source image — fidelity to the input | `clip`, `dinov2`, `pixel` |
| `pr` | k-NN precision / recall ([Kynkäänniemi et al., 2019](https://arxiv.org/abs/1904.06991)) | `dinov2` |

Each row of the output CSV is one grid cell (one parameter combination) with a
column per requested `type × extractor`, so you can rank settings or plot a
metric against any swept axis (e.g. content similarity vs. gate steepness `nu`).

##  References
This repository was heavily inspired by the following repositories:
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://github.com/NVlabs/edm/tree/main)
- [EDM2 and Autoguidance](https://github.com/NVlabs/edm2)

##  Acknowledgements


This work was done during an internship at the Generative Memory Lab under the supervision of Luca Ambrogioni. I would also like to thank Dejan Stančević for the many discussions.

