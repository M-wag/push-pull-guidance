"""
HTML viewer generator for sweep manifests.

Two viewing modes:
  - Grid mode: pick two axes for rows/columns, filter the rest
  - Single mode: one image with dropdowns/sliders to navigate
"""

import json
import os
import shutil
from pathlib import Path


def build_viewer_html(manifest, base_dir, example_paths=None, prompts=None, title="Sweep Viewer"):
    """
    Build an HTML viewer that references images by relative path.

    manifest: dict loaded from manifest.json
    base_dir: directory containing the images/ subdirectory
    example_paths: optional list of example image file paths
    prompts: optional list of prompt strings
    """
    axes = manifest["axes"]  # {name: [values]}
    entries = manifest["entries"]

    # Use relative paths directly (browser loads from disk)
    data_entries = []
    for entry in entries:
        logs_rel = entry.get("logs")
        logs_plot = logs_rel.replace("logs.json", "logs_plot.png") if logs_rel else None
        data_entries.append({
            "params":        entry["params"],
            "images":        entry["images"],
            "snapshots_dir": entry.get("snapshots"),
            "logs_plot":     logs_plot,
        })

    # Copy example images into output dir so they're accessible via relative paths
    example_rel_paths = []
    if example_paths:
        examples_dir = os.path.join(base_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)
        for i, p in enumerate(example_paths):
            if os.path.exists(p):
                ext = os.path.splitext(p)[1] or ".png"
                dst = os.path.join(examples_dir, f"example_{i}{ext}")
                if not os.path.exists(dst):
                    shutil.copy2(p, dst)
                example_rel_paths.append(f"examples/example_{i}{ext}")

    js_data = json.dumps({
        "axes":           axes,
        "entries":        data_entries,
        "examples":       example_rel_paths,
        "prompts":        prompts or [],
        "fixed":          manifest.get("fixed", {}),
        "snapshot_steps": manifest.get("snapshot_steps", []),
    })

    return _HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", js_data).replace("__TITLE__", title)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
:root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --text: #e0e0e0;
    --text-muted: #a0a0b0;
    --accent: #e94560;
    --border: #2a2a4a;
    --control-bg: #0f3460;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 1.5rem 2rem;
    line-height: 1.5;
}
h1 {
    font-size: 1.6rem;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}
h2 {
    font-size: 1.2rem;
    color: var(--accent);
    margin: 1.2rem 0 0.6rem;
}

/* Mode tabs */
.tabs {
    display: flex;
    gap: 0;
    margin-bottom: 1rem;
}
.tab {
    padding: 0.5rem 1.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    cursor: pointer;
    color: var(--text-muted);
    font-size: 0.9rem;
}
.tab:first-child { border-radius: 4px 0 0 4px; }
.tab:last-child { border-radius: 0 4px 4px 0; }
.tab.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
}

/* Controls bar */
.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    align-items: center;
    margin-bottom: 1rem;
    padding: 0.8rem 1rem;
    background: var(--surface);
    border-radius: 6px;
}
.control-group {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.control-group label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.control-group select, .control-group input[type="range"] {
    background: var(--control-bg);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 0.3rem 0.5rem;
    border-radius: 3px;
    font-size: 0.85rem;
}
.control-group select { min-width: 120px; }
.control-group input[type="range"] { min-width: 140px; }
.slider-value {
    font-size: 0.8rem;
    color: var(--accent);
    text-align: center;
}

/* Image sample selector */
.sample-selector {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
    align-items: center;
}
.sample-btn {
    padding: 0.3rem 0.8rem;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text-muted);
    cursor: pointer;
    border-radius: 3px;
    font-size: 0.85rem;
}
.sample-btn.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
}

/* Grid mode */
#grid-view table {
    border-collapse: collapse;
    margin: 0 auto;
}
#grid-view th, #grid-view td {
    border: 1px solid var(--border);
    padding: 0.4rem;
    text-align: center;
    vertical-align: middle;
}
#grid-view th {
    background: var(--surface);
    font-weight: 600;
    font-size: 0.85rem;
    white-space: nowrap;
}
#grid-view td img {
    max-width: 200px;
    border-radius: 3px;
}
.row-label {
    font-weight: 600;
    text-align: right;
    padding-right: 0.8rem !important;
    white-space: nowrap;
    font-size: 0.85rem;
}

/* Single mode */
#single-view {
    display: flex;
    gap: 2rem;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
}
.single-card {
    text-align: center;
}
.single-card img {
    max-width: 400px;
    border-radius: 4px;
    border: 1px solid var(--border);
}
.single-card .caption {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.3rem;
}
.single-card .label {
    font-size: 0.75rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}

/* Panel toggles */
.panel-toggles {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-bottom: 0.8rem;
    flex-wrap: wrap;
}
.panel-toggles span {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-right: 0.3rem;
}
.toggle-btn {
    padding: 0.25rem 0.8rem;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-muted);
    cursor: pointer;
    font-size: 0.8rem;
    user-select: none;
    transition: background 0.15s, color 0.15s;
}
.toggle-btn.on {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
}

/* Fixed params */
.fixed-params {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 1rem;
}
.fixed-params span {
    background: var(--surface);
    padding: 0.15rem 0.5rem;
    border-radius: 3px;
    margin-right: 0.5rem;
}
</style>
</head>
<body>

<h1>__TITLE__</h1>
<div id="fixed-params" class="fixed-params"></div>

<!-- Mode tabs -->
<div class="tabs">
    <div class="tab active" onclick="setMode('grid')">Grid</div>
    <div class="tab" onclick="setMode('single')">Single</div>
</div>

<!-- Controls -->
<div id="controls" class="controls"></div>

<!-- Panel toggles (single mode only) -->
<div id="panel-toggles" class="panel-toggles" style="display:none;"></div>

<!-- Sample selector -->
<div id="sample-selector" class="sample-selector"></div>

<!-- Views -->
<div id="grid-view"></div>
<div id="single-view" style="display:none;"></div>

<script>
const DATA = __DATA_PLACEHOLDER__;

const axisNames = Object.keys(DATA.axes);
const axisValues = DATA.axes;
const entries = DATA.entries;
const examples = DATA.examples;
const prompts = DATA.prompts;
const snapshotSteps = DATA.snapshot_steps || [];   // e.g. [0, 10, 25, 49]
const hasSnapshots = snapshotSteps.length > 0;
const nImages = entries.length > 0 ? entries[0].images.length : 0;

let mode = "grid";
let currentSample = 0;
let currentStepIdx = snapshotSteps.length > 0 ? snapshotSteps.length - 1 : -1; // -1 = final image

// Which panels are visible in single mode
const panels = {
    example:     { label: "Example",     on: true },
    generated:   { label: "Generated",   on: true },
    diagnostics: { label: "Diagnostics", on: true },
    timeline:    { label: "Timeline",    on: true },
};

// State
let gridRowAxis = axisNames[0] || "";
let gridColAxis = axisNames[1] || axisNames[0] || "";
let filters = {};
let singleSelections = {};

axisNames.forEach(name => {
    filters[name] = axisValues[name][0];
    singleSelections[name] = axisValues[name][0];
});

// Fixed params banner
const fixedDiv = document.getElementById("fixed-params");
const fixedParts = Object.entries(DATA.fixed).map(([k, v]) => `<span>${k}: ${v}</span>`);
if (fixedParts.length) fixedDiv.innerHTML = "Fixed: " + fixedParts.join("");

function lookupEntry(params) {
    return entries.find(e =>
        Object.keys(params).every(k => String(e.params[k]) === String(params[k]))
    );
}

// Return the correct image path for an entry given the current step selection.
// stepIdx == -1 means "final image"; otherwise index into snapshotSteps.
function imageForStep(entry, sampleIdx, stepIdx) {
    if (stepIdx >= 0 && entry && entry.snapshots_dir) {
        return `${entry.snapshots_dir}/img_${sampleIdx}_step_${stepIdx}.png`;
    }
    return entry ? entry.images[sampleIdx] : null;
}

function setMode(m) {
    mode = m;
    document.querySelectorAll(".tab").forEach((t, i) => {
        t.classList.toggle("active", (i === 0 && m === "grid") || (i === 1 && m === "single"));
    });
    document.getElementById("grid-view").style.display = m === "grid" ? "" : "none";
    document.getElementById("single-view").style.display = m === "single" ? "" : "none";
    document.getElementById("panel-toggles").style.display = m === "single" ? "" : "none";
    buildControls();
    buildPanelToggles();
    render();
}

function buildPanelToggles() {
    const container = document.getElementById("panel-toggles");
    let html = '<span>Show:</span>';
    Object.entries(panels).forEach(([key, p]) => {
        html += `<div class="toggle-btn ${p.on ? 'on' : ''}" onclick="togglePanel('${key}')">${p.label}</div>`;
    });
    container.innerHTML = html;
}

function togglePanel(key) {
    panels[key].on = !panels[key].on;
    buildPanelToggles();
    renderSingle();
}

function buildSampleSelector() {
    const container = document.getElementById("sample-selector");
    if (nImages <= 1) { container.innerHTML = ""; return; }
    let html = '<span style="font-size:0.85rem;color:var(--text-muted)">Sample:</span>';
    for (let i = 0; i < nImages; i++) {
        const label = prompts[i] ? `${i}: ${prompts[i].substring(0, 30)}` : `#${i}`;
        html += `<div class="sample-btn ${i === currentSample ? 'active' : ''}" onclick="setSample(${i})">${label}</div>`;
    }
    container.innerHTML = html;
}

function setSample(i) {
    currentSample = i;
    buildSampleSelector();
    render();
}

// Step label: "Final" for -1, otherwise "Step N"
function stepLabel(idx) {
    return idx < 0 ? "Final" : `Step ${snapshotSteps[idx]}`;
}

function buildStepSlider() {
    if (!hasSnapshots) return "";
    // Values: -1 (final) then 0..S-1 (snapshot steps)
    const total = snapshotSteps.length + 1;  // +1 for final
    const sliderVal = currentStepIdx < 0 ? snapshotSteps.length : currentStepIdx;
    return `<div class="control-group">
        <label>Denoising Step</label>
        <input type="range" id="step-slider" min="0" max="${snapshotSteps.length}"
            value="${sliderVal}" oninput="setStepIdx(this.value)">
        <div class="slider-value" id="step-label">${stepLabel(currentStepIdx)}</div>
    </div>`;
}

function setStepIdx(val) {
    const v = parseInt(val);
    // slider max = snapshotSteps.length means "final"
    currentStepIdx = (v >= snapshotSteps.length) ? -1 : v;
    const el = document.getElementById("step-label");
    if (el) el.textContent = stepLabel(currentStepIdx);
    render();
}

function buildControls() {
    const container = document.getElementById("controls");
    let html = "";

    if (mode === "grid") {
        html += buildSelect("Rows", "grid-row", axisNames, gridRowAxis, "setGridRow(this.value)");
        html += buildSelect("Columns", "grid-col", axisNames, gridColAxis, "setGridCol(this.value)");
        axisNames.forEach(name => {
            if (name !== gridRowAxis && name !== gridColAxis) {
                html += buildSelect(name, `filter-${name}`, axisValues[name], filters[name],
                    `setFilter('${name}', this.value)`);
            }
        });
        html += buildStepSlider();
    } else {
        axisNames.forEach(name => {
            const vals = axisValues[name];
            const allNumeric = vals.every(v => typeof v === "number" || (typeof v === "string" && v !== "" && !isNaN(v)));
            if (allNumeric && vals.length > 2) {
                html += buildSlider(name, vals, singleSelections[name]);
            } else {
                html += buildSelect(name, `single-${name}`, vals, singleSelections[name],
                    `setSingleVal('${name}', this.value)`);
            }
        });
    }
    container.innerHTML = html;
}

function buildSelect(label, id, options, selected, onchange) {
    let html = `<div class="control-group"><label>${label}</label><select id="${id}" onchange="${onchange}">`;
    options.forEach(v => {
        html += `<option value="${v}" ${String(v) === String(selected) ? 'selected' : ''}>${v}</option>`;
    });
    html += "</select></div>";
    return html;
}

function buildSlider(label, values, selected) {
    const idx = values.findIndex(v => String(v) === String(selected));
    return `<div class="control-group">
        <label>${label}</label>
        <input type="range" min="0" max="${values.length - 1}" value="${idx >= 0 ? idx : 0}"
            oninput="setSingleSlider('${label}', this.value, ${JSON.stringify(values)})">
        <div class="slider-value" id="slider-val-${label}">${selected}</div>
    </div>`;
}

function setGridRow(v) { gridRowAxis = v; if (gridColAxis === v) gridColAxis = axisNames.find(n => n !== v) || v; buildControls(); render(); }
function setGridCol(v) { gridColAxis = v; if (gridRowAxis === v) gridRowAxis = axisNames.find(n => n !== v) || v; buildControls(); render(); }
function setFilter(axis, v) { filters[axis] = v; render(); }
function setSingleVal(axis, v) { singleSelections[axis] = isNaN(v) ? v : (v.includes('.') ? parseFloat(v) : parseInt(v)); render(); }
function setSingleSlider(axis, idx, values) {
    singleSelections[axis] = values[idx];
    const el = document.getElementById(`slider-val-${axis}`);
    if (el) el.textContent = values[idx];
    render();
}

function render() {
    buildSampleSelector();
    if (mode === "grid") renderGrid();
    else renderSingle();
}

function renderGrid() {
    const container = document.getElementById("grid-view");
    const rowVals = axisValues[gridRowAxis];
    const colVals = axisValues[gridColAxis];

    const filterParams = {};
    axisNames.forEach(name => {
        if (name !== gridRowAxis && name !== gridColAxis) filterParams[name] = filters[name];
    });

    let html = "<table><thead><tr><th></th>";
    colVals.forEach(cv => { html += `<th>${gridColAxis}=${cv}</th>`; });
    html += "</tr></thead><tbody>";

    rowVals.forEach(rv => {
        html += `<tr><td class="row-label">${gridRowAxis}=${rv}</td>`;
        colVals.forEach(cv => {
            const params = { ...filterParams, [gridRowAxis]: rv, [gridColAxis]: cv };
            const entry = lookupEntry(params);
            const src = imageForStep(entry, currentSample, currentStepIdx);
            if (src) {
                html += `<td><img src="${src}"></td>`;
            } else {
                html += `<td style="color:var(--text-muted);font-size:0.8rem">—</td>`;
            }
        });
        html += "</tr>";
    });
    html += "</tbody></table>";
    container.innerHTML = html;
}

function renderSingle() {
    const container = document.getElementById("single-view");
    const params = {};
    axisNames.forEach(name => { params[name] = singleSelections[name]; });
    const entry = lookupEntry(params);

    let html = "";

    // Example
    if (panels.example.on && examples[currentSample]) {
        html += `<div class="single-card">
            <div class="label">Example</div>
            <img src="${examples[currentSample]}">
            ${prompts[currentSample] ? `<div class="caption">${prompts[currentSample]}</div>` : ""}
        </div>`;
    }

    // Final generated image
    if (panels.generated.on) {
        html += `<div class="single-card"><div class="label">Generated</div>`;
        const finalSrc = entry ? entry.images[currentSample] : null;
        if (finalSrc) {
            html += `<img src="${finalSrc}">`;
        } else {
            html += `<div style="padding:4rem;color:var(--text-muted)">No image for these parameters</div>`;
        }
        const paramStr = Object.entries(params).map(([k, v]) => `${k}=${v}`).join(", ");
        html += `<div class="caption">${paramStr}</div>`;
        if (prompts[currentSample]) html += `<div class="caption">${prompts[currentSample]}</div>`;
        html += "</div>";
    }

    // Score diagnostics plot
    if (panels.diagnostics.on && entry && entry.logs_plot) {
        html += `<div style="flex-basis:100%;margin-top:0.5rem;">
            <div class="label" style="margin-bottom:0.4rem;">Score Diagnostics</div>
            <img src="${entry.logs_plot}" style="width:100%;border:1px solid var(--border);border-radius:4px;">
        </div>`;
    }

    // Denoising timeline (only if this entry has snapshots)
    if (panels.timeline.on && hasSnapshots && entry && entry.snapshots_dir) {
        const snapSrc = (idx) => `${entry.snapshots_dir}/img_${currentSample}_step_${idx}.png`;
        const initIdx = 0;
        html += `<div class="single-card" style="min-width:300px;">
            <div class="label">Denoising Timeline</div>
            <img id="snap-img" src="${snapSrc(initIdx)}" style="max-width:400px;border:1px solid var(--border);border-radius:4px;">
            <div style="margin-top:0.6rem;">
                <input type="range" min="0" max="${snapshotSteps.length - 1}" value="${initIdx}"
                    style="width:100%;accent-color:var(--accent);"
                    oninput="updateSnapImg(this.value, '${entry.snapshots_dir}')">
                <div id="snap-label" style="font-size:0.8rem;color:var(--accent);text-align:center;margin-top:0.2rem;">
                    Step ${snapshotSteps[initIdx]}
                </div>
            </div>
        </div>`;
    }

    container.innerHTML = html;
}

function updateSnapImg(idx, snapDir) {
    const i = parseInt(idx);
    const img = document.getElementById("snap-img");
    if (img) img.src = `${snapDir}/img_${currentSample}_step_${i}.png`;
    const lbl = document.getElementById("snap-label");
    if (lbl) lbl.textContent = `Step ${snapshotSteps[i]}`;
}

// Init
buildControls();
buildPanelToggles();
render();
</script>
</body>
</html>"""
