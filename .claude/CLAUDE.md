# push-pull-guidance

Diffusion-based image editing research. Runs parameter sweeps, produces images and
metrics, and renders an HTML viewer over the results.

- **`docs/architecture.md`** — the target design: layering, contracts, and why each
  boundary sits where it does. Read before any structural change.
- **`docs/migration.md`** — how to get there from here, in nine steps that each leave
  the repo runnable, plus the delegation checklist.

**Migration status: step 0 not started.** <!-- keep this line current -->

## Invariants

These hold in the target design. Breaking one is a bug, not a style preference.

- `ppg_sweep/core/` must never import from `ppg_sweep/diffusion/`. Domain specifics
  reach core as data or as protocol implementations, never as imports.
- A cell is identified by `cell_id`, the hash of its resolved config — never by its
  ordinal position in the grid.
- Nothing outside `core/paths.py` and the sinks constructs a file path. The viewer's
  JavaScript builds none at all: every path it needs is a resolved string in the manifest.
- The manifest is derived. Delete it and rebuild it losslessly from the plan, the
  receipts and the event log. It is never the source of truth for any value.
- `Engine.build()` must fully reset every field it might set, or state leaks between cells.

## Inherited code — do not restyle or refactor

- `ppg/` — git submodule, the PPG algorithm itself. Separate repo.
- `edm/` — NVlabs edm / edm2. Pinned, unmodified.
- `torch_utils/` — NVlabs, **locally modified**. It deliberately differs from
  `edm/torch_utils/` (imports `dnnlib` rather than `util`; `training_stats` calls
  commented out). Both copies are needed. Neither is stale.
- `calculate_metrics.py` — NVlabs, heavily forked. The CLIP and pixel detectors,
  `CSTransform`, `PRTransform` and `calculate_metrics_from_iterable` are local additions.

## Known bugs the refactor exists to fix

Do not paper over these.

- `metrics_sweep.py` ignores `examples.samples_yaml` and scores a randomly sampled
  image set, while `sweep.py` uses the curated one (`sweep.py:791` vs
  `metrics_sweep.py:145`). Same config, two experiments.
- The metrics CSV resumes on `cell_idx`, so resuming after any config edit silently
  mixes results from different parameter settings into one table.
- `SweepRunner.build()` returns early when a cell has no `ppg` block, leaving
  `normalize_variance` and `use_net` holding the previous cell's values. Latent today
  because no single run mixes cells with and without `ppg`.
- `Gallery._cell_complete` is dead code — defined, never called.

## Commands

```bash
python sweep.py demo/config.yaml                        # sweep + HTML viewer
torchrun --nproc-per-node=4 sweep.py demo/config.yaml   # multi-GPU
python metrics_sweep.py demo/config.yaml                # metrics CSV
python sweep.py demo/config.yaml --build-html           # rebuild viewer only
```

## Conventions

- Keep pure-move commits separate from behaviour-change commits. Never mix `git mv`
  with edits in one commit.
- Every refactor step must leave the repo runnable and pass the step-0 golden plan
  file byte-identically unless the change to it was intentional.
