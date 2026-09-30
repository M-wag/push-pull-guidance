# PPG Sweep — Migration Plan

How to get from the current repo to the design in [`architecture.md`](architecture.md), in steps that each leave the repo runnable.

## How the current repo differs

The structural gap is that **`sweep.py` is secretly a library**. `metrics_sweep.py` imports nine names from it, including `SweepRunner`, `PIPELINE_SETUP` and the dataset loaders, and calls the private `runner._make_inputs()`. Importing it also pulls in matplotlib and executes `matplotlib.use("Agg")` as a side effect. Roughly 100 lines of `main()` are duplicated between the two files.

Where each piece of today's code lands:

| Today | Target | Change |
| --- | --- | --- |
| `sweeper/schema.py` | `config/schema.py` | Moved as-is; add the metrics fields |
| `load_sweep_config` + `_load_metrics_sweep_config` | `config/loader.py` | Merged into one |
| `sweeper/grid.py` | `config/grid.py` | Add `cell_id` hashing |
| — | `plan.py` | New; plus provenance capture |
| `setup_pipeline_*`, `MODEL_DEFAULTS` | `model/pipeline.py` | Moved out of `sweep.py` |
| `build_map_layers`, `ProjectionCache` | `model/maps.py` | Moved out of `sweep.py` |
| `load_imgnet64`, `load_imgnet_qualitative`, `load_wildti2i` | `model/datasets.py` | Moved; one caller |
| `Gallery.generate` + `metrics_sweep.main` loop | `run/runner.py` | Two loops become one |
| `Gallery._save_*`, CSV append | `run/sinks.py` | Extracted from the loops |
| `Gallery` manifest logic | `store/manifest.py` | Resolved paths, `cell_id`, metrics, failures |
| `sweeper/viewer.py` (720 lines) | `viewer/*` + `viz/spec.py` | CSS and JS become real files |
| `make_log_plots` | `viz/figures.py` + `plots.js` | Two renderers over logs.json; drop the dark theme |

The two central classes need splitting at method granularity, because neither is a runner in the sense used above. `SweepRunner` assembles models, loads data, builds inputs, generates, and accumulates. `Gallery` iterates, stores, catalogues, and builds HTML. Nine concerns across two objects, divided along a line that does not correspond to any layer boundary.

| Today | What it actually does | Target home |
| --- | --- | --- |
| `SweepRunner.build()` | assemble PPG, maps and solver from a cell | `model/pipeline.py` |
| `SweepRunner._get_example_tensors()` | load and VAE-encode source images | `model/datasets.py`, once at setup |
| `SweepRunner._make_inputs()` | build the input iterable stack | `model/pipeline.py` |
| `SweepRunner.run()` — driving | step the solver over batches | `generate(cell)` |
| `SweepRunner.run()` — collecting | accumulate images and snapshots in memory | `ImageSink` |
| `SweepRunner._collect_result()` | average batch logs | `LogSink` |
| `Gallery.generate()` — loop | iterate cells, check resume | `run/runner.py` |
| `Gallery._save_images/_snapshots/_logs` | write files | the sinks |
| `Gallery` manifest methods | record what exists | `store/manifest.py` |
| `Gallery.build_html()` | render the viewer | `viz/`, not the runner's concern |

So the runner sketch does not delete `SweepRunner` — its `model.build(...)` and `generate(cell)` lines *are* `SweepRunner`, narrowed to model assembly and execution. The two lines worth noticing are the split of `run()`: driving the solver stays in the model layer, accumulating results moves to a sink. That separation is what makes streaming `on_batch` possible, and it is why metrics never need all 450 images in memory.

Four defects this fixes, each currently live:

1. **The dataset divergence.** `sweep.py:791` branches on `examples_cfg["samples_yaml"]` and calls `load_imgnet_qualitative`. `metrics_sweep.py:145` unconditionally calls `load_imgnet64`. So `metrics_sweep.py demo/config.yaml` silently ignores `samples_yaml` and scores a *different, randomly sampled* set of images than the viewer displays. Same config, two experiments.
2. **Resume corruption.** Covered above: the CSV keys on ordinal index, so resuming after a config edit mixes parameter settings within one table.
3. **The cross-language path convention.** `gallery._save_snapshots` writes `snapshots/img_{i}_step_{s}.png`; `viewer.js` rebuilds that filename with string interpolation. Rename in Python, and the JS fails as a missing-image icon with no error.
4. **Unvalidated metrics config.** `_load_metrics_sweep_config` pops `ref_stats`, `example_features_dir` and friends off the raw dict and returns them untyped. A typo becomes `None` and the run proceeds.

Two smaller things worth carrying along. `RAISE_ERRORS` and `SEED_BY_DATASET_INDEX` are module-level globals in `sweep.py` that silently change behaviour; they belong in the config. And nothing currently records the repo's git SHA, the `ppg` submodule's SHA, or the checkpoint used — which is the thing you will most regret in six months, and costs ten lines in `plan.py`.

Documentation has already drifted, which is the argument for keeping this document next to the code: `diagnostics/design.md` describes an older design (an explicit `axes:` block, base64-embedded images) that no longer matches the code, and `metrics_sweep.py`'s docstring points at a `plan.md` that does not exist.

## Migration path

**Refactor from the leaves inward, and do the runner last.** The runner is the most connected component in the design: it touches the engine, the sinks, the plan and the store. Narrowing it first means inventing all four of those interfaces at once, which is exactly the rewrite this plan exists to avoid. Left until the end, it becomes a small and obvious diff, because by then everything it talks to already has a shape.

Order by dependency depth, shallowest first:

| Component | Depends on | GPU to test? |
| --- | --- | --- |
| Config loader | nothing | no |
| Plan + `cell_id` | config | no |
| Store: layout, manifest | plan | no |
| Viewer | the manifest format only | no |
| Sinks | store | no |
| Engine | config + torch | yes |
| Runner | engine + sinks + plan | yes |

The viewer splits across that ordering. Its mechanical half — getting CSS and JS out of a Python string — depends on nothing and can be picked up whenever you want a break from the risky work. Wiring it to the new manifest cannot begin until the manifest builder exists.

**The technique for every step: build the new thing beside the old, then make the old code call into it.** Never delete and replace in one commit. The old entry points keep working throughout, and each step is revertible on its own.

A second rule that saves real pain: **keep pure-move commits separate from behaviour-change commits.** `git mv` alone in one commit, edits in the next. A diff that both moves and changes a file is unreviewable, and you will want to review these.

### Step 0 — the safety net (about a day)

Write a script that dumps, for each of the four demo configs, the full list of `(cell index, coords, hash of resolved config)` to a golden file. Commit it. Every later step must leave that file byte-identical unless you intended otherwise.

This is the highest-value half-day in the project. The central risk of the whole refactor is silently changing *which cells run or with what parameters*, and this catches exactly that, in about thirty lines, with no model and no GPU.

**The golden file is only half of it.** It answers "am I running the same cells with the same parameters" and says nothing about whether the images are still right. Step 2 could restructure the schema, produce an identical plan, and still attach the gate with the wrong scale. Only a real run sees that.

So also build a **smoke config** small enough to run on every commit: `n_entries: 2`, `n_seeds: 1`, `num_steps: 4`, three rungs of `nu` instead of twenty-two, snapshots at two steps. That is 3 cells × 2 images × 4 steps ≈ 24 network evaluations — seconds on any GPU. Two variants: a pixel one that runs always, and a pushpull one exercising the interpolation map, the VAE and the numdiff pullback, run before merging anything that touches maps.

It asserts three things, and the second is the one that gets forgotten:

1. **Images match reference within tolerance.** At 64×64 the references are a few KB, so commit them. Try exact hashes first — the churn-seeding work suggests your pipeline is close to deterministic — and fall back to `np.abs(a - b).max() < tol` only if cuDNN algorithm choice or TF32 makes them flaky.
2. **The manifest matches a reference manifest**, with `origin` stripped. This is what catches sinks writing to the wrong place, a missing receipt, or a cell reported complete when it is not — none of which shows up in the pixels.
3. **Metric values match within tolerance**, when the metrics sink ran.

One scenario beyond the happy path is worth scripting: **resume**. Run the smoke sweep, kill it after one cell, re-run, assert it finishes with a correct manifest. That is exactly where step 3 and step 4 bugs live, and it has the worst blast radius of any of them.

Capture the reference images **before step 1**, from current code. Step 2 will change the config format, so the smoke config gets rewritten there — but the reference images should come out identical, since only the spelling changed. That is not an inconvenience; it is the strongest test in the plan, and the thing that proves the schema restructure preserved semantics.

### Step 1 — one config loader

Add the metrics fields to the pydantic schema and make `load_sweep_config` handle both shapes. Then reduce the old function to a shim that returns the same tuple it used to:

```python
def _load_metrics_sweep_config(path):
    cfg = load_sweep_config(path)
    return cfg, cfg.metrics_extras()
```

Nothing else in `metrics_sweep.py` changes. Small diff, fixes the live `samples_yaml` divergence, and gives you the habit of the shim pattern on the easiest possible case.

### Step 2 — restructure the schema

Introduce the `engine:` block with `solver` and the method union nested inside it, and the method registry: `pushpull`, `pixel`, `sdedit`, `none`. `MODEL_DEFAULTS` and `noise_source` move under the engine at the same time. Type the `examples` dict while you are here — it is the untyped hole that caused the divergence step 1 patched over.

**This has to precede cell ids.** A `cell_id` is a hash of the resolved config, so any later change to the config's shape invalidates every id already computed. Doing the schema work first means hashing once, against the final shape.

It also removes the stale-state bug by construction: with "no guidance" as a method rather than an absent `ppg` block, `build()` no longer has an early return that leaves `normalize_variance` holding the previous cell's value.

### Step 3 — cell ids, the plan, and the baseline

Add hashing and `ExperimentPlan` with its six groups. Then — the part that de-risks it — **add `cell_id` alongside the existing keys rather than replacing them.** A new CSV column, a new manifest field; directories keep their ordinal names and resume still uses the old key.

Run a sweep. Confirm the ids are stable across reruns and unchanged by reordering a config's keys. Only then flip resume over in a second, tiny commit.

Fold the baseline in here: as a cell with no guidance it gets an id like any other, and `_run_baseline` with its `baseline_meta.json` caching goes away.

### Step 4 — sinks and receipts

Write `ImageSink`, `SnapshotSink`, `LogSink` and `MetricsSink`, each returning a receipt. Then change `Gallery.generate` so that where it calls `self._save_images(...)` it calls the sink instead, and do the same inside `metrics_sweep`'s loop. **Leave both loops, their signatures and their callers exactly as they are.**

After this the storage code exists once, behind an interface, exercised by both paths — and receipts are being written, which the next step needs.

### Step 5 — the manifest builder

Write `core/manifest.py`: plan plus receipts plus event log in, `manifest.json` out, including failed cells and their reasons. Add the group layer here too — `compare.group`, the `example_set_hash` check, and the per-cell detail files.

`merge_manifests` disappears at this point, because each rank's receipts already live in its own cell directories.

### Step 6 — the viewer

Two halves with different dependencies. Pulling the CSS and JS out of `viewer.py` into real files is mechanical and needs nothing. Rewriting `data.js` against the new manifest, and adding `plots.js` for the interactive plots, cannot start until step 5 is done.

### Step 7 — the engine, by subtraction

Give `SweepRunner` the `Engine` shape without moving anything: it already has `build()`, and `run()` is close enough to `generate()`. Then remove responsibilities one commit at a time, using the delegation inventory as the checklist — dataset loading out to setup, accumulation out to `ImageSink`, log averaging out to `LogSink`. Each removal is a few lines and independently revertible.

### Step 8 — the runner, which is now easy

Both loops build an engine and dispatch to sinks, and look nearly identical. Deleting one and sharing the other is a small diff, and it is the point at which generating each image twice stops.

Steps 1 to 3 remove the three ways this codebase can currently produce quietly wrong numbers — the dataset divergence, the stale guidance state, and the CSV resume corruption — and step 3 is a prerequisite for everything after it. Steps 5 and 6 are what the three yes-answers cost: cross-run comparison, interactive plots and metrics in the viewer are new capability rather than cleanup, and between them they are the bulk of the work. Step 8 is not merely cost-of-change — `sweep.py` and `metrics_sweep.py` currently generate the same 9,900 images per run independently, so sinks halve the GPU time of any run that wants both.

If the preprint is close: do 0 to 3, then step 5 only far enough to get the metric-against-strength numbers out as a CSV that `figures.py` can plot. The interactive viewer is worth building once the figures are settled, not while they are still moving.

### On the directory move

Do not do it as a big-bang first step. A large no-op diff touching every import costs you a merge-conflict surface and a broken `git blame`, and reduces nothing.

Instead: create `ppg_sweep/core/` and `ppg_sweep/diffusion/` empty at step 0, put each new module into whichever half it belongs to as you write it, and move an old file only when a step is rewriting it anyway. The inherited code — the `ppg` submodule, `edm/`, `torch_utils/`, `calculate_metrics.py` — moves last and all at once, since none of it is changing; that commit needs `.gitmodules` repointed and a find-and-replace across imports, and nothing else. The restructure then arrives as a consequence of the work rather than as a prelude to it.

## Delegation inventory

Everything the current sweep drivers do that belongs somewhere else, grouped by destination. The runner keeps only the six things listed under Contracts; this is the remainder, spread today across `SweepRunner`, `Gallery` and the loop in `metrics_sweep.main`.

**To the Engine (`model/`)**

- [ ] PPG assembly — `create_sgdm`, the map-config walk, `create_ppg_composed` (`SweepRunner.build`)
- [ ] Solver mutation — `sigma_max`, `apply_2nd_order`, `solver_seed`, stochastic presets
- [ ] `MODEL_DEFAULTS` lookup and the `max_batch_size` override
- [ ] `latent_dim` / `noise_shape` bookkeeping threaded through the map chain
- [ ] Input iterable construction (`_make_inputs`), including the `sdedit_mix` closure
- [ ] `noise_source` branching: random / ddim\_inversion / sdedit
- [ ] `MetadataIterable` injection for `cs` metrics, which currently mutates `inputs.extensions` from outside
- [ ] Logger installation driven by `config.logging.mode`
- [ ] `gc.collect()` and `torch.cuda.empty_cache()`

**To setup, run once (`model/datasets.py`)**

- [ ] Example image loading and VAE encoding (`_get_example_tensors`)
- [ ] DDIM inversion precompute and its distributed broadcast — 26 duplicated lines

**To the Plan (`plan.py`)**

- [ ] Axis extraction and grid iteration (`extract_axes`, `_grid`)
- [ ] `unflatten(flat_cell, config)` per cell — precompute instead
- [ ] `n_images`, currently a property branching on model type
- [ ] `snapshot_steps` resolution from the config dict

**To sinks (`run/sinks.py`)**

- [ ] Image accumulation (`all_images[idx] = img`)
- [ ] Snapshot accumulation (`all_snapshots[idx][s]`)
- [ ] Array-to-PIL conversion: dtype scaling, CHW to HWC, channel squeeze (`_arr_to_pil`)
- [ ] PNG writing (`_save_images`, `_save_snapshots`)
- [ ] Per-batch log collection and `logger.reset()`
- [ ] Batch-log averaging across batches (`_collect_result`)
- [ ] `logs.json` writing (`_save_logs`)
- [ ] Per-cell config dump (`_save_cell_config`)
- [ ] CSV row assembly and append (`_append_csv_row`)
- [ ] Resume predicates — manifest lookup, `_read_done_indices`

**To the store (`store/`)**

- [ ] Cell directory naming (`_cell_dir`)
- [ ] Relative path computation for manifest entries
- [ ] Manifest load, create, save and validate — four methods
- [ ] Stale-entry pruning (`_validate_manifest`)
- [ ] Per-rank manifest paths and merging (`_manifest_path_for_rank`, `merge_manifests`)
- [ ] Failure records (`_log_computed_cell` writing `computed_cells.jsonl`), which become a status field on the cell

**To the viewer and figures**

- [ ] Plot orchestration (`_generate_plots`)
- [ ] HTML building, example copying, baseline path reconstruction (`build_html`)
- [ ] Persisting `prompts` and `n_seeds` into the manifest as viewer metadata

**Delete rather than move**

- [ ] `Gallery._cell_complete` — defined and never called
- [ ] `RAISE_ERRORS`, `SEED_BY_DATASET_INDEX` — module globals that belong in the config

The test for whether this is finished is the one under Contracts: the runner imports neither torch nor anything that writes files.
