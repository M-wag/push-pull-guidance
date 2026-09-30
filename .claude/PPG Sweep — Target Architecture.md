# PPG Sweep — Target Architecture

Sep 22, 2026 · @Brandon

## Purpose

This document defines the target architecture for the `push-pull-guidance` sweep and visualisation stack, and the route from the current repo to it.

The target in one paragraph: one resolver turns `config.yaml` into an **ExperimentPlan** of content-addressed cells. One runner walks that plan and streams each cell's output to a set of **sinks** (images, metrics, logs). Everything written lands under a `cell_id`. One **manifest** file describes what exists on disk, and the viewer reads only that manifest — never the filesystem layout.

The design constraint throughout is that this is a single-author research repo with a preprint pending. It optimises for two things: being unable to silently produce wrong results, and being legible to its author six months from now. It does not optimise for scale, extensibility, or multiple contributors.

Decided here: cell identity, the four durable artifacts, layer boundaries, module layout, and migration order. Left open: the three questions in the final section, which change the design and still need answers.

## Vocabulary

Three words below are load-bearing, and none of them are standard ML terms. All three name code that already exists in this repo — it is just not separated out.

**Runner** — the `for` loop over parameter settings. It picks the next cell, decides whether to skip it, and asks the model for images. It never writes a file. Today this loop is written twice: `Gallery.generate` and the body of `metrics_sweep.main`.

**Sink** — something the loop hands each result to, which decides what to store and where. Today this code sits *inside* those two loops: `_save_images`, `_save_snapshots`, `_save_logs` and the manifest append in one; `_append_csv_row` in the other. The name is borrowed from stream processing (source → transform → sink) and nothing hangs on it.

Separating the two means writing the loop once and choosing sinks per invocation:

```python
run(plan, sinks=[ImageSink(out)])                    # what sweep.py does today
run(plan, sinks=[MetricsSink(csv)])                  # what metrics_sweep.py does today
run(plan, sinks=[ImageSink(out), MetricsSink(csv)])  # both, generating each image once
```

**Manifest** — a catalogue of what was produced and where it sits, so that no reader has to know the directory layout. Today this is `manifest.json`, doing about half the job. To display one snapshot the viewer currently has to know four separate facts: that images live in the cell directory, that snapshots sit in a `snapshots/` subdirectory, that they are named `img_{i}_step_{s}.png`, and that `i = example_idx × n_seeds + seed_idx`. Those four facts are defined in Python and re-implemented in JavaScript. With a complete manifest the viewer looks up one string and knows none of them.

## Design principles

Five rules generate every other decision in this document. Where a later section seems arbitrary, it is downstream of one of these.

1. **One resolver, many consumers.** `config.yaml` is parsed and validated in exactly one place. Anything that needs to know what the experiment is reads the resolved plan, never the YAML. Two parsers always drift; this repo already has two that have.
2. **Identity is content, not position.** A cell is named by a hash of its resolved parameters, not by its ordinal index in the Cartesian product. Adding one value to one axis must not renumber anything.
3. **The manifest is the interface.** Producers write files and register them in a manifest. Consumers read the manifest. No consumer reconstructs a path from a naming convention, because a convention known in two languages is a convention that will break silently.
4. **Describe what exists, not what was requested.** The viewer's capabilities are derived from the manifest, not declared in the config. Cells fail, runs get interrupted, metrics arrive later. A spec written from intent will promise panels that have no data.
5. **Data is the durable artifact; renderings are disposable.** Images, metric rows and step logs are kept. Plots, HTML and figures are regenerated from them at will, by whichever renderer suits the destination.

## The spine: cell identity

A **cell** is one point in the parameter grid, and `cell_id` is the hash of its resolved configuration. This single decision is what lets heterogeneous storage — PNGs, a CSV, JSON logs — behave as one dataset.

Today a cell is identified by its ordinal index in the Cartesian product. Directories are named `0007_decomposed_6.46_inf`; the metrics CSV keys rows on `cell_idx`. Add one value to one `values:` list and every index after it shifts. The manifest survives this because `_validate_manifest` re-keys on `flat_cell`, but the CSV does not: `_read_done_indices` reads `cell_idx`, so resuming a metrics run after any config edit mixes results from different parameter settings into one table.

A content hash fixes this and buys three things at once: exact resume, config edits that invalidate only the cells that actually changed, and a **join key** — images and metrics for the same cell line up for free, with no extra plumbing.

Four artifacts hang off that spine. Each has one writer and a stable shape.

| Artifact | Written by | Read by | Holds |
| --- | --- | --- | --- |
| Config | the user | resolver | The YAML sweep definition, unchanged from today |
| Plan | resolver | runner, manifest builder | Resolved cells, axis order, provenance |
| Results | sinks | viewer, figures, analysis | PNGs, metric rows, step logs, all under `cell_id` |
| Manifest | manifest builder | viewer, figures | What exists, where, and with every path resolved |

Identity has **three levels**, because the viewer compares across runs as well as within one.

```python
@dataclass(frozen=True)
class Cell:
    run_id:   str            # which sweep: pushpull | pixel | sdedit
    cell_id:  str            # hash of resolved config, unique within a run
    coords:   dict           # swept values only — grid layout in the viewer
    resolved: SweepConfig    # complete config — what the model runs at

@dataclass(frozen=True)
class ExampleKey:
    example_id: str          # stable across runs: source image + source/target label
    seed:       int
```

`coords` has to exist separately from `resolved`. The viewer's row and column selectors need to know which parameters vary and in what order, and cannot recover that from a pile of complete configs.

`ExampleKey` is the join axis **across** runs, as `cell_id` is the join axis within one. Today an image is `img_{i}.png` where `i = example_idx * n_seeds + seed_idx`, and `example_idx` is just a position in `my_samples.yaml`. That happens to line up across the three survey runs because their `examples` blocks are identical, but nothing enforces it — change `n_entries` in one config and every cross-run comparison silently shifts by one image. The concept already exists in the metrics path as `MetadataIterable(class_id, example_id)`; it needs promoting to the storage key.

Storage therefore keys on `runs/<run_id>/cells/<cell_id>/`, with images named by `ExampleKey` rather than by ordinal position.

**The baseline is a cell.** It is the configuration with guidance removed, so giving it a `cell_id` like any other deletes `_run_baseline` — roughly forty lines of bespoke caching with its own `baseline_meta.json`, duplicating what resume already does. It then flows through the sinks and lands in the manifest for free.

It also answers a question you cannot currently check. The three survey arms share one `baseline_dir`, and if their baseline configurations really are identical they hash alike and are computed once. If they do not hash alike, that is worth knowing before the arms are compared.

## Cross-run comparison

The three survey runs sweep **the same 22-value ladder bound to a different parameter each time**. That correspondence is semantic, nothing can infer it, and so it has to be declared.

| Run | Strength knob | Path in config | Other differences |
| --- | --- | --- | --- |
| `pixel` | guidance strength | `ppg.gate.nu` | quadratic gate, no maps |
| `pushpull` | guidance strength | `ppg.gate.nu` | heaviside gate, interpolation + VAE maps |
| `sdedit` | noise level | `solver.sigma_max` | no `ppg` block at all |

Everything else is already matched by hand: identical `examples` blocks (`my_samples.yaml`, `n_entries: 50`, `n_seeds: 9`, `seed: 0`), identical `snapshots.steps`, identical solver settings, one shared `baseline_dir`, and the identical 22-element value list `[80.0, 66.93, ..., 0.6, 0.0]`.

So the axes are **not** unionable. `ppg.gate.nu` and `solver.sigma_max` are the same conceptual axis under two names, and `sdedit` has no `ppg` block to align on. A small declaration in each config carries what the paths cannot:

```yaml
compare:
  group: survey             # runs in one group are comparable
  arm: sdedit               # this run's label in comparison views
  axes:
    strength: solver.sigma_max    # pixel/pushpull say: ppg.gate.nu
```

Three views fall out of that, and they are the reason to do any of this:

1. **Matched-strength image row.** Fix an `ExampleKey` and a rung of the ladder; show baseline, pixel, pushpull and sdedit side by side. This is the qualitative figure the paper wants.
2. **Metric against strength, one line per arm.** Content similarity or FD on the y-axis, the shared `strength` axis on the x, one line per run. This is the quantitative figure the paper wants, and it is impossible without both the alias and the `cell_id` join.
3. **Per-step diagnostics overlaid within a run.** Score norms across all 22 rungs on one chart, which no collection of per-cell PNGs can show.

**The comparability contract.** The manifest records a hash of the example set (`example_id` list plus `n_seeds` plus seed) per run. The viewer overlays two runs only when their hashes match, and says so plainly when they do not. Without this, changing `n_entries` in one config produces a comparison chart that looks fine and is wrong — the visual analogue of the CSV resume bug, and harder to notice.

## Target architecture

Five layers, each depending only on the one below it. The rule that matters: **the arrows never point backwards**, and nothing skips a layer to reach the filesystem.

```mermaid
flowchart TD
    A["config.yaml × 3 arms"] --> B["Resolver<br/>schema + grid + ids"]
    B --> C["ExperimentPlan<br/>cells, examples, provenance"]
    C --> D["Engine<br/>built once: net, data, caches"]
    C --> R["Runner<br/>loop, resume, dispatch"]
    D --> R
    R --> S["Sinks<br/>images / metrics / logs"]
    S --> F[("Results<br/>PNGs, csv, logs")]
    S --> P[("Receipts<br/>what each sink wrote")]
    R --> L[("Event log<br/>attempts + errors")]
    C --> G["Manifest builder"]
    P --> G
    L --> G
    G --> H[("manifest.json<br/>+ per-cell detail")]
    H --> I["Viewer (JS)"]
    H --> J["figures.py"]
    F -.-> I
```

The **resolver** owns the schema and imports no torch. The **plan** is pure data — cells, the example set, provenance — so building or inspecting it needs no GPU, which is what makes `--dry-run` and the comparability hash cheap. The **engine** holds everything expensive that is built once: the network, encoded example tensors, the DDIM precompute, the projection caches. The **runner** owns the loop and knows nothing about what happens to output. The **sinks** own storage and know nothing about the grid. The **manifest builder** reads the plan, the receipts and the event log — never the results directory itself — and writes the catalogue. The **viewer** and `figures.py` are peers: two renderers over one manifest, neither aware of the other. The dotted edge is the viewer fetching the actual images, at URLs the manifest gave it.

The important separations, stated as prohibitions:

- The **model layer** (pipeline setup, maps, dynamics) never touches the filesystem or the config's output paths. It receives a `resolved` config and returns images.
- The **runner** never writes a file. It emits events.
- The **viewer** never constructs a path. Every path it needs is a literal string in the manifest.
- The **config** never declares viewer panels. Panels are derived from what the manifest contains.

That last one deserves emphasis because it is the least obvious. The viewer already does this correctly for one case — `baseline: { on: baselines.length > 0 }` — and hardcodes the rest. The target simply applies the existing pattern to timeline, diagnostics and metrics, computed in Python at manifest-build time.

## Module layout

One importable package replaces the loose top-level scripts. This is a net simplification: the repo currently has `sweep.py`, `generate.py`, `util.py` and `calculate_metrics.py` at the root, plus `sweeper/`, `diagnostics/`, `torch_utils/` and `edm/`.

```
ppg_sweep/
  config/
    schema.py       # pydantic models — unchanged from sweeper/schema.py
    loader.py       # THE loader: yaml -> Config, metrics fields included
    grid.py         # axes -> cells, cell_id hashing
  plan.py           # Config -> ExperimentPlan: cells, example set, origin
  model/
    engine.py       # build_engine(plan) -> Engine — all one-time setup
    pipeline.py     # sd / edm / edm2 construction
    maps.py         # build_map_layers + ProjectionCache
    datasets.py     # WHICH examples: paths, labels, seeds — no torch
  run/
    runner.py       # the loop: resume, dispatch, error handling
    sinks.py        # ImageSink, SnapshotSink, LogSink, MetricsSink
  store/
    paths.py        # THE path conventions — single source of truth
    manifest.py     # build + read the manifest
  viz/
    spec.py         # manifest -> presentation spec
    figures.py      # matplotlib renderer for the paper
  cli.py            # run | metrics | manifest | view

viewer/
  index.html
  viewer.js         # UI
  data.js           # the ONLY module that knows the manifest shape
  plots.js          # numbers -> interactive plots
  viewer.css
```

Three notes on the boundaries.

**There is no separate setup layer — setup is the engine's constructor.** `build_engine(plan)` does everything currently scattered across the two `main()` functions: loads the network, resolves and encodes the example tensors, runs the DDIM precompute and its broadcast, and warms the projection cache. That keeps the concept count down and puts the expensive one-time work in the object that holds its results.

The split between `datasets.py` and `engine.py` follows the torch boundary. *Which* examples a run uses — paths, class labels, prompts, seeds — is plan data, resolved without a GPU, so the comparability hash and `n_images` can be computed from a config alone. *Loading* those images and encoding them to latents is engine state. Today both halves live in `SweepRunner` and `main()` together, which is why neither can be inspected without CUDA.

`store/paths.py` is the only module that builds a path. Both Python and JS currently encode the snapshot naming convention independently; after this, Python encodes it once and JS not at all.

`viewer/` becomes real files. Today `sweeper/viewer.py` is 720 lines, of which roughly 217 are CSS and 402 are JavaScript inside a Python string with a `__DATA_PLACEHOLDER__` substitution. Extracting them costs nothing and buys linting, editor support and diffs that mean something.

`sweep.py` and `metrics_sweep.py` stay as three-line shims that call into `cli.py`, so the commands in the README keep working.

## Contracts

Five interfaces carry the whole design. Everything else is implementation.

**Resolver.** One entry point, and it is the only code that reads YAML.

```python
def load(path: str) -> Config: ...           # validates everything, metrics fields included
def plan(cfg: Config) -> ExperimentPlan: ...  # no torch import, no side effects
```

`plan()` being pure and cheap means `--dry-run` falls out for free: print the 36 cells and their resolved configs without touching a GPU. That is a useful check when nested variant axes make the grid hard to predict.

**ExperimentPlan.** The resolver's output, and the keystone of the design. It is also a *persisted* artifact, written to `plan.json` when a run starts — which is what lets `ppg manifest` rebuild a catalogue days later from `plan.json` plus the results directory, without re-reading the config or needing it to still say the same thing.

Its contents are best derived from its consumers rather than from the config's shape:

| Consumer | Needs from the plan |
| --- | --- |
| Runner | the ordered cells, and nothing else |
| `build_engine` | model and checkpoint, example set, seeds, batch size, snapshot steps, logging mode |
| Sinks | `cell_id`, the `ExampleKey`s, the output root |
| Manifest builder | every cell including those that never ran, axes in declared order, aliases, provenance |
| Viewer, via the manifest | axis order, aliases, arm label, title |
| `--dry-run` and the golden test | all of it, deterministically serialised |
| Comparability check | `example_set_hash`, before anything runs |

Group those fields by **role** rather than listing them flat, and separate what every cell has in common from what the sweep moves. A flat plan hides both distinctions, and it makes every cell carry a full resolved config — 22 near-identical copies of the model block, the maps and the solver.

```python
@dataclass(frozen=True)
class ExperimentPlan:
    identification: Identification  # run_id, group, arm, title
    shared:         CommonConfig    # the config every cell has in common
    varied:         Sweep           # axes, axis_aliases, cells
    data:           DataPlan        # examples, n_seeds, example_set_hash
    origin:         Origin          # git shas, checkpoint, created_at
    paths:          Paths           # output root, baseline dir

    def resolved(self, cell: Cell) -> CommonConfig:
        """shared merged with cell.coords — derived, never stored."""

@dataclass(frozen=True)
class Cell:
    cell_id: str                    # hash of the resolved config
    coords:  dict                   # the varied values, and nothing else
```

One caveat carried over from the current schema: `CommonConfig` is the same pydantic model whether it holds `Axis` objects (straight from the loader), the shared remainder, or a cell's fully concrete config, because swept fields are typed `Union[float, Axis]`. The type cannot tell you which of the three you have. Splitting it into a spec type and a concrete type would fix that; it is not worth the two extra models until something actually goes wrong.

`unflatten()` already performs the `shared` + `coords` merge today; the difference is that the shared half becomes a field of the plan rather than an argument threaded through the loop. That is what lets the engine be handed the common config directly at setup, and what turns `resolved` into a method instead of 22 stored copies of the same model block.

The grouping pays for itself at the call sites, because each consumer takes only the slice it needs:

| Consumer | Receives |
| --- | --- |
| `build_engine` | `shared`, `data`, `paths` |
| Runner | `varied`, plus the engine and sinks |
| Sinks | `paths` and the cell ids |
| Manifest builder | the whole plan |

That is what turns the layering from a convention into something a type checker enforces. The runner cannot reach into `provenance` to stamp a filename, because it was never handed it.

Four invariants keep it useful:

1. **It imports no torch.** Otherwise `--dry-run`, the golden test and the comparability check all need a GPU. This is what forces the `datasets.py` split: *which* examples is plan data, *loading* them is engine state.
2. **It is frozen after construction.** If the runner could mutate it, `cell_id` would stop meaning anything.
3. **It serialises deterministically.** Sorted keys and the infinity sentinel, or hashes differ between processes.
4. **It is closed over the whole grid.** A cell that fails is still in the plan — that is how the manifest can report it as missing rather than silently omitting it.

What stays out is as important. Anything that depends on execution — status, timings, metric values — belongs to the results. Loaded tensors and the network belong to the engine. Display defaults belong to the viewer spec, derived from the manifest. The one judgement call is `axis_aliases`, which look like presentation but are not: asserting that `sdedit`'s `sigma_max` and `pushpull`'s `nu` are the same quantity is a claim about the experiment, not about the display.

**Engine.** The model layer's face to the runner, and the reason the runner stays small. It is constructed once at setup, with the network loaded, datasets resolved and caches warm.

```python
class Engine(Protocol):
    def build(self, resolved: SweepConfig) -> None: ...      # configure for this cell
    def generate(self, resolved: SweepConfig) -> Iterator[Batch]: ...
```

The engine is stateful on purpose — reloading the network or rebuilding projection matrices per cell would be ruinous, which is what `ProjectionCache` already exists to avoid. The cost of that choice is a hard requirement: **`build()` must fully reset every field it might set.** Today it does not. `SweepRunner.build()` sets `dynamics.ppg = None` and returns early when a cell has no `ppg` block, leaving `normalize_variance` and `use_net` at the previous cell's values. It does not currently fire, because `pixel` and `pushpull` set `ppg` on every cell and `sdedit` sets it on none — but a variant axis that makes `ppg` optional within one run would silently contaminate every cell after the first.

With the engine defined, the runner owns exactly six things and nothing else: iteration order, distributed work-splitting, the skip decision (delegated to the sinks as a predicate), calling build then generate, fanning batches out to sinks, and per-cell error handling. Roughly forty lines.

The test that keeps it honest: **the runner must be unit-testable with a fake engine and fake sinks, importing neither torch nor anything that writes files.** If a change makes that impossible, the change put something in the wrong layer.

**Sink.** The runner hands each cell's output to a list of these. A sink is just "what to do with the result", pulled out of the loop.

```python
class Sink(Protocol):
    def should_skip(self, cell: Cell) -> bool: ...        # resume logic lives here
    def on_batch(self, cell: Cell, batch) -> None: ...    # streaming
    def on_cell_end(self, cell: Cell) -> Receipt: ...     # what it wrote, and where
```

The interface is streaming rather than "here is the finished list" because the two existing consumers differ: metrics reduce across the distributed group as batches arrive and never hold all images, while the viewer path collects everything in memory. Streaming serves both, and `ImageSink` accumulates internally.

The runner's skip rule follows from having several sinks: **skip a cell only when every sink says skip.** When one sink wants a cell another already holds — metrics missing, images present — the cell runs, and the sinks that consider it done still receive its batches. They must tolerate that, by ignoring them or by overwriting harmlessly.

The payoff beyond deduplication: today every image is generated **twice**, once under `sweep.py` for the viewer and once under `metrics_sweep.py` for the table. With sinks you generate once and feed both.

**Receipts.** `on_cell_end` returns a receipt naming every file that sink wrote and where it put it; the runner collects them into `cells/<cell_id>/receipt.json`. This is what stops the manifest builder needing to know the directory layout — it reads the plan, the receipts and the event log, and never joins a path or stats a file.

It is the manifest pattern applied one level down. The manifest exists so the viewer need not know filesystem conventions; receipts exist so the builder need not either. The coupling does not disappear, it moves: from two modules independently knowing a naming scheme, to one small declared format.

The alternative is to let the builder depend on `store/paths.py` and probe — ask where cell X's snapshot for example key Y *would* live, then test whether it is there. That keeps one source of truth, but it is \~119,000 `stat` calls per run, painful on a network filesystem, and blind to anything the conventions do not predict. `store/paths.py` still owns those conventions for the sinks' benefit; the builder reaches for it only in an opt-in `--rescan` repair mode, for when receipts are missing or suspect.

Two consequences fall out:

- **`merge_manifests` disappears.** Under `torchrun` each rank owns a disjoint set of cells and writes receipts into its own cell directories. No contention, no per-rank manifest files, nothing to stitch afterwards — the builder was always going to merge N receipts.
- **A cell killed mid-write has no receipt**, so the builder reports it incomplete rather than trusting whatever files happen to be on disk. That is the correct reading, and it comes for free.

**Manifest.** Two tiers, because one file does not survive the real scale. A survey run is 22 cells × 450 images = 9,900 finals, plus 11 snapshot steps each = 108,900 more. That is \~119k PNGs per run and \~356k across the three. Fully resolving every path into one file gives roughly 6 MB per run and 18 MB for the group — too much to load up front, and almost all of it never looked at.

So: a small top manifest holding everything needed to *navigate and plot*, and a per-cell detail file holding the image and snapshot paths, fetched when the user opens that cell (\~270 KB each).

```json
// manifest.json — one per group, loaded up front
{
  "group": "survey",
  "runs": [{
    "run_id": "pushpull",
    "arm": "pushpull",
    "provenance": {"git_sha": "7838308", "ppg_sha": "...", "created": "..."},
    "example_set_hash": "9c41e2",
    "axes": {"ppg.gate.nu": [80.0, 66.93, "...", 0.0]},
    "axis_aliases": {"strength": "ppg.gate.nu"},
    "cells": [{
      "cell_id": "a3f2c1d8",
      "coords": {"ppg.gate.nu": 6.46},
      "status": "ok",
      "detail": "runs/pushpull/cells/a3f2c1d8/cell.json",
      "metrics": {"fd_dinov2": 12.4, "cs_clip": 0.81},
      "log_summary": {"norm_guide": [0.92, 0.88, "..."]}
    }]
  }],
  "examples": [{"example_id": "237-21", "image": "examples/237-21.png", "prompt": "..."}],
  "baselines": {"237-21:0": "baselines/237-21_seed0.png"}
}
```

Three things earn their place in the top tier. `metrics` are scalars, so all 22 cells × 3 runs cost almost nothing and make the metric-against-strength chart instant. `log_summary` is the batch-mean curve — 32 steps × \~6 series × 66 cells ≈ 150 KB — which makes the overlay plot instant too. `example_set_hash` is what the comparability contract checks.

```json
// runs/pushpull/cells/a3f2c1d8/cell.json — fetched on navigation
{
  "images":    {"237-21:0": "img/237-21_seed0.png"},
  "snapshots": {"237-21:0": ["snap/237-21_seed0_step0.png", "..."]},
  "logs":      "logs.json"
}
```

Snapshots stay a resolved list rather than a directory. That is what deletes the JS line rebuilding filenames — the single most breakable thing in the current viewer — and the tiering preserves it, since Python still resolves every path and JS still builds none.

**Data access (JS).** One module, and the rest of the viewer goes through it.

```js
listRuns(group)                              // -> arms, axes, aliases
listCells(runId, filter)                     // -> cells matching coords
getImage(runId, cellId, exampleKey, step)    // -> url, lazily loads cell.json
getMetric(runId, cellId, name)               // -> number, from the top manifest
getStrengthSeries(group, metric)             // -> one series per arm, for plots.js
```

`data.js` is the only file that knows the manifest shape, so lazy loading, caching and any future server change live in one place.

One consequence of the scale: **`file://` stops being the delivery mechanism.** Browsers block `fetch()` on `file://` URLs, so per-cell lazy loading needs a server, and at \~1 GB per group nobody is emailing this folder anyway. The answer is `ppg view <group_dir>`, which serves the directory statically and opens a browser; over SSH it is one `-L` port-forward. If a self-contained single run must stay openable from disk, the fallback is writing the data as `.js` assignment files loaded by `<script>` tags, which are not subject to the same restriction.

## Data flow

One invocation of `ppg run demo/config.yaml --metrics`, from YAML to rendered viewer.

```mermaid
sequenceDiagram
    participant CLI
    participant Resolver
    participant Runner
    participant Engine
    participant Sinks
    participant Manifest
    CLI->>Resolver: load + plan(config.yaml)
    Resolver-->>CLI: ExperimentPlan (22 cells)
    CLI->>Runner: run(plan, engine, sinks)
    loop per cell
        Runner->>Sinks: should_skip(cell)?
        Sinks-->>Runner: skip only if all agree
        Runner->>Engine: build(cell.resolved)
        Engine-->>Runner: batches of images
        Runner->>Sinks: on_batch(cell, batch)
        Runner->>Sinks: on_cell_end(cell)
        Sinks-->>Runner: Receipt
        Runner->>Runner: write receipt + event log
    end
    CLI->>Manifest: build(plan, receipts, log)
    Manifest-->>CLI: manifest.json
```

The loop is the same whether you asked for images, metrics, or both — only the sink list differs. `ppg run` passes `[ImageSink, SnapshotSink, LogSink]`; `ppg metrics` passes `[MetricsSink]`; `--metrics` passes all four and generates each image once.

Viewing is a separate, cheap step that never touches a GPU:

1. `ppg manifest <run_dir>` reads `plan.json`, the per-cell receipts and the event log, and writes `manifest.json` — including the cells that failed, with the reason.
2. `viz/spec.py` reads the manifest and derives the presentation spec — which panels have data, which axes vary, sensible defaults for rows and columns.
3. `viewer/index.html` loads `manifest.json` plus the spec and renders. No build step and no application server — `ppg view` just serves the directory.

Because steps 1–3 are decoupled from generation, a partially finished sweep is viewable, and a re-render after changing the CSS costs nothing.

## Code ownership

Roughly half this repo is NVlabs code carried over from `edm` and `edm2`, and it should be treated as a vendored dependency rather than as material to restructure. The copyright headers draw the line cleanly:

| Area | Origin | Treatment |
| --- | --- | --- |
| `edm/`, `torch_utils/` | NVIDIA headers | Vendor: one copy under `vendor/`, pinned, never restyled |
| `calculate_metrics.py` | NVIDIA header, heavily extended | Split — see below |
| `generate.py`, `util.py` | no header, EDM-influenced | Yours; good design, just move into the package |
| `sweep.py`, `metrics_sweep.py`, `sweeper/`, `diagnostics/` | yours | Free to restructure |

Two consequences worth acting on.

`torch_utils/` at the top level duplicates `edm/torch_utils/` and carries the same NVIDIA headers, so it is a stray copy of vendored code rather than a module of yours. One copy under `vendor/` settles the 8-line drift between them.

`calculate_metrics.py` is the awkward case: 1,008 lines under an NVIDIA header, but the CLIP and pixel detectors, `CSTransform`, `PRTransform` and `calculate_metrics_from_iterable` are all additions. A heavily forked vendored file is the worst maintenance case, because upstream fixes can no longer be pulled in and the boundary between inherited and original code is invisible to a reader. Worth splitting: the detector and statistics machinery stays vendored, and your metric types move into the package.

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

The viewer depends on nothing but the manifest format, so it is a parallel track you can pick up whenever you want a break from the risky work.

**The technique for every step: build the new thing beside the old, then make the old code call into it.** Never delete and replace in one commit. The old entry points keep working throughout, and each step is revertible on its own.

A second rule that saves real pain: **keep pure-move commits separate from behaviour-change commits.** `git mv` alone in one commit, edits in the next. A diff that both moves and changes a file is unreviewable, and you will want to review these.

### Step 0 — the safety net (half a day, no GPU)

Write a script that dumps, for each of the four demo configs, the full list of `(cell index, coords, hash of resolved config)` to a golden file. Commit it. Every later step must leave that file byte-identical unless you intended otherwise.

This is the highest-value half-day in the project. The central risk of the whole refactor is silently changing *which cells run or with what parameters*, and this catches exactly that, in about thirty lines, with no model and no GPU.

### Step 1 — one config loader

Add the metrics fields to the pydantic schema and make `load_sweep_config` handle both shapes. Then reduce the old function to a shim that returns the same tuple it used to:

```python
def _load_metrics_sweep_config(path):
    cfg = load_sweep_config(path)
    return cfg, cfg.metrics_extras()
```

Nothing else in `metrics_sweep.py` changes. Small diff, fixes the live `samples_yaml` divergence, and gives you the habit of the shim pattern on the easiest possible case.

### Step 2 — add `cell_id` without depending on it

Build `plan.py` and hash the resolved configs. Then — and this is the part that de-risks it — **add `cell_id` alongside the existing keys rather than replacing them.** A new column in the CSV, a new field in each manifest entry. Directories keep their ordinal names. Resume still uses the old key.

Run a sweep. Confirm the ids are stable, that re-running reproduces them, and that reordering a config's keys does not change them. Only then flip resume over to `cell_id` in a second, tiny commit. Renaming the directories can wait until step 5 or never.

### Step 3 — extract the viewer (parallel track)

Turn the CSS and JS strings into real files that get copied next to the generated HTML, and add `data.js` so the JS stops constructing paths. Same manifest in, same page out. No model runs, nothing else depends on it, and it converts the most tedious file in the repo into one you can actually edit.

### Step 4 — sinks, underneath the existing loop

Write `ImageSink`, `SnapshotSink`, `LogSink` and `MetricsSink` as classes. Then change `Gallery.generate` so that where it calls `self._save_images(...)` it calls `sink.on_cell_end(...)` instead. **Leave the loop, its signature and its callers exactly as they are.** Do the same inside `metrics_sweep`'s loop for `MetricsSink`.

After this the storage code exists once, behind an interface, and is exercised by both paths — but you have not yet touched control flow.

### Step 5 — the engine, by subtraction

Give `SweepRunner` the `Engine` shape without moving anything: it already has `build()`, and `run()` is close enough to `generate()`. Then remove responsibilities from it one commit at a time, using the delegation inventory as the checklist — dataset loading out to setup, accumulation out to `ImageSink`, log averaging out to `LogSink`. Each removal is a few lines and independently revertible.

### Step 6 — the runner, which is now easy

By this point both loops build an engine and dispatch to sinks, and they look nearly identical. Deleting one and sharing the other is a small diff, and it is the point at which generating each image twice stops.

### On the directory move

Do not do it as a big-bang first step. A large no-op diff touching every import costs you a merge-conflict surface and a broken `git blame`, and reduces nothing.

Instead: create `ppg_sweep/` empty at step 0, put each new module there as you write it, and move an old file only when a step is rewriting it anyway. Whatever is still at the top level after step 6 moves in one small commit at the end. The restructure then arrives as a consequence of the work rather than as a prelude to it.

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

**To viz (`viz/`)**

- [ ] Plot orchestration (`_generate_plots`)
- [ ] HTML building, example copying, baseline path reconstruction (`build_html`)
- [ ] Persisting `prompts` and `n_seeds` into the manifest as viewer metadata

**Delete rather than move**

- [ ] `Gallery._cell_complete` — defined and never called
- [ ] `RAISE_ERRORS`, `SEED_BY_DATASET_INDEX` — module globals that belong in the config

The test for whether this is finished is the one under Contracts: the runner imports neither torch nor anything that writes files.

## Non-goals

This design refuses the following on purpose. Each is a reasonable idea at a larger scale and a liability at this one.

- **Code generation from the config.** It buys nothing that validated typed objects do not, and costs a build step and unreadable stack traces. The plan is data, not generated code.
- **A backend API or database.** `ppg view` serves the directory as static files; it does not query, filter or compute. The manifest is built ahead of time and the server only hands over bytes. Revisit if manifest build time becomes the bottleneck — SQLite behind the same `data.js` interface is the first thing to reach for then, not a request-handling backend.
- **A shared plot-spec abstraction.** An earlier draft of this design proposed one spec rendered by both matplotlib and JS. That is over-engineered for a solo project. Two independent renderers reading the same `logs.json` is simpler and loses nothing.
- **A plugin or registry system for sinks and models.** A list of sinks passed to a function is enough. There will not be third-party sinks.
- **Abstracting over "any diffusion model".** The three cases in `PIPELINE_SETUP` are the three cases. Generalise on the fourth, if it arrives.
- **Touching `edm/`, `generate.py`, or the pydantic schema's semantics.** `generate.py`'s iterable-composition design is good and the schema is good. Leave them.
- **`diagnostics/`.** Nine one-off scripts, each parsing YAML by hand. Out of scope; let them adopt the new loader opportunistically.

One deferred cleanup worth noting rather than doing: `torch_utils/` and `edm/torch_utils/` are near-duplicates, differing by 8 lines each in `misc.py` and `distributed.py`. Worth reconciling eventually, unrelated to this refactor.

## Decisions taken and what remains

The three open questions are answered, all in the maximal direction: the viewer compares across runs, some plots are interactive, and metrics appear in the viewer. Their consequences are folded into the sections above — the three-level identity, the cross-run section, and the two-tier manifest.

What those answers cost, so the scope is not a surprise:

| Answer | Consequence |
| --- | --- |
| Compare across runs | `run_id` + `ExampleKey`; declared axis aliases; comparability hash; the manifest becomes a group registry |
| Interactive plots | `logs.json` shipped to the browser; `plots.js`; batch-mean curves promoted into the top manifest |
| Metrics in the viewer | Metric scalars in the top manifest; the CSV becomes an export rather than the source of truth |
| All three together | A local server (`ppg view`) replaces opening the HTML from disk |

Four smaller things remain open:

1. **Which metric leads.** The metric-against-strength chart needs a default y-axis. `cs_dinov2` and `fd_dinov2` measure opposing things, and the genuinely useful view may be one plotted against the other, with strength as the path traced through the plane.
2. **What the bottom of the ladder means across arms.** All three configs end at `0.0`, but zero guidance strength and zero noise are not the same limit. Worth confirming both reduce to the baseline before putting them on a shared x-axis.
3. **Whether snapshots can be thinned.** 108,900 snapshot PNGs per run dominate disk, manifest size and build time, and 11 steps × 450 images per cell is more than the viewer will ever show. Sampling snapshots for a subset of examples would cut the largest single cost in the system.
4. **The `ppg` submodule is SSH-only.** `git@github.com:M-wag/ppg.git` fails for anyone following the README's `--recurse-submodules` instruction without access. Unrelated to this architecture, but it blocks reproduction of the preprint.
