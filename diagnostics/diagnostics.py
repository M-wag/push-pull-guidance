from __future__ import annotations

"""
Lightweight HTML diagnostics report generator.

Usage:
    from diagnostics_report import DiagnosticsReport

    report = DiagnosticsReport("My Experiment")
    report.add_header("Section 1")
    report.add_note("Some observation with $\\LaTeX$ support")
    report.add_image_row([pil_img1, pil_img2], captions=["A", "B"])
    report.save("diagnostics.html")

Supports: PIL Images, numpy arrays (uint8 or float [0,1]), and file paths.
"""

import base64
import io
from pathlib import Path
from typing import Union
from datetime import datetime

try:
    from PIL import Image
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    from PIL import Image
    HAS_NUMPY = False


ImageLike = Union[Image.Image, str, Path, "np.ndarray"]


def _to_base64(img: ImageLike, fmt: str = "png") -> str:
    """Convert an image to a base64-encoded data URI."""
    if isinstance(img, (str, Path)):
        img = Image.open(img)

    # Matplotlib Figure -> render to PNG bytes directly
    try:
        from matplotlib.figure import Figure
        if isinstance(img, Figure):
            buf = io.BytesIO()
            img.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/{fmt};base64,{b64}"
    except ImportError:
        pass

    if HAS_NUMPY and isinstance(img, np.ndarray):
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img.clip(0, 1) * 255).astype(np.uint8)
        # Handle CHW format (e.g. from PyTorch tensors) -> HWC
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[2] not in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))
        if img.ndim == 3 and img.shape[2] == 1:
            img = img.squeeze(2)
        img = Image.fromarray(img)

    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/{fmt};base64,{b64}"


class DiagnosticsReport:
    def __init__(self, title: str = "Diagnostics Report"):
        self.title = title
        self._blocks: list[str] = []

    # ── Content methods ──────────────────────────────────────────────

    def add_header(self, text: str, level: int = 2):
        """Add a section header (h2 by default)."""
        tag = f"h{max(1, min(6, level))}"
        self._blocks.append(f"<{tag}>{text}</{tag}>")

    def add_note(self, text: str):
        """Add a paragraph of text. Supports inline LaTeX ($...$) and display LaTeX ($$...$$)."""
        self._blocks.append(f'<p class="note">{text}</p>')

    def add_image(self, img: ImageLike, caption: str = "", width: str = "auto"):
        """Add a single image with an optional caption."""
        uri = _to_base64(img)
        html = f'<figure><img src="{uri}" style="max-width:{width};"><figcaption>{caption}</figcaption></figure>'
        self._blocks.append(html)

    def add_image_row(self, images: list[ImageLike], captions: list[str] | None = None):
        """Add a row of images displayed side by side."""
        captions = captions or [""] * len(images)
        assert len(captions) == len(images), "Number of captions must match number of images"
        cells = []
        for img, cap in zip(images, captions):
            uri = _to_base64(img)
            cells.append(
                f'<figure class="row-item">'
                f'<img src="{uri}">'
                f'<figcaption>{cap}</figcaption>'
                f'</figure>'
            )
        self._blocks.append(f'<div class="image-row">{"".join(cells)}</div>')

    def add_image_grid(self, images: list[list[ImageLike]],
                       row_labels: list[str] | None = None,
                       col_labels: list[str] | None = None):
        """Add a grid of images as an HTML table. Useful for method × sample comparisons."""
        html = '<table class="image-grid"><thead><tr>'
        if row_labels:
            html += "<th></th>"
        if col_labels:
            for cl in col_labels:
                html += f"<th>{cl}</th>"
        html += "</tr></thead><tbody>"
        for i, row in enumerate(images):
            html += "<tr>"
            if row_labels:
                html += f'<td class="row-label">{row_labels[i]}</td>'
            for img in row:
                uri = _to_base64(img)
                html += f'<td><img src="{uri}"></td>'
            html += "</tr>"
        html += "</tbody></table>"
        self._blocks.append(html)

    _slider_count = 0  # class-level counter for unique IDs

    def add_image_slider(self, images: list[list[ImageLike]],
                         row_labels: list[str] | None = None,
                         step_labels: list[str] | None = None):
        """Add a slider that scrubs through columns of images.

        images: list of rows, each row is a list of images (one per step).
                All rows must have the same number of steps.
        row_labels: label for each row (displayed to the left).
        step_labels: label for each step (shown above the slider).
        """
        n_steps = len(images[0])
        assert all(len(row) == n_steps for row in images), "All rows must have the same number of steps"
        step_labels = step_labels or [str(i) for i in range(n_steps)]
        row_labels = row_labels or [f"Row {i}" for i in range(len(images))]

        uid = f"slider_{DiagnosticsReport._slider_count}"
        DiagnosticsReport._slider_count += 1

        # Encode all images
        uris = [[_to_base64(img) for img in row] for row in images]

        # Build HTML: one visible image per row, a slider, and JS to swap src
        rows_html = ""
        for r, (label, row_uris) in enumerate(zip(row_labels, uris)):
            rows_html += (
                f'<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">'
                f'<span class="row-label" style="min-width:80px;text-align:right;">{label}</span>'
                f'<img id="{uid}_r{r}" src="{row_uris[0]}" '
                f'style="max-width:256px;border-radius:4px;border:1px solid var(--border);">'
                f'</div>'
            )

        # JSON array of URI arrays for JS
        import json
        uris_json = json.dumps(uris)

        labels_json = json.dumps(step_labels)

        html = f"""
<div class="image-slider-widget" style="margin:1rem 0;">
{rows_html}
<div style="display:flex;align-items:center;gap:1rem;margin-top:0.8rem;">
  <input type="range" id="{uid}_range" min="0" max="{n_steps - 1}" value="0"
         style="flex:1;accent-color:var(--accent);">
  <span id="{uid}_label" style="min-width:80px;font-size:0.9rem;">{step_labels[0]}</span>
</div>
<script>
(function() {{
  var uris = {uris_json};
  var labels = {labels_json};
  var slider = document.getElementById("{uid}_range");
  var labelEl = document.getElementById("{uid}_label");
  slider.addEventListener("input", function() {{
    var idx = parseInt(this.value);
    labelEl.textContent = labels[idx];
    for (var r = 0; r < uris.length; r++) {{
      document.getElementById("{uid}_r" + r).src = uris[r][idx];
    }}
  }});
}})();
</script>
</div>
"""
        self._blocks.append(html)

    def add_table(self, headers: list[str], rows: list[list[str]]):
        """Add a data table (e.g. for metrics). Cell contents support LaTeX."""
        html = "<table><thead><tr>"
        for h in headers:
            html += f"<th>{h}</th>"
        html += "</tr></thead><tbody>"
        for row in rows:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</tbody></table>"
        self._blocks.append(html)

    def add_separator(self):
        self._blocks.append("<hr>")

    def add_html(self, raw: str):
        """Inject arbitrary HTML for anything not covered above."""
        self._blocks.append(raw)

    ## Rendering ##

    def _render(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = "\n".join(self._blocks)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{self.title}</title>

<!-- KaTeX for LaTeX rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$', right: '$', display: false}}
        ]
    }});"></script>

<style>
    :root {{
        --bg: #1a1a2e;
        --surface: #16213e;
        --text: #e0e0e0;
        --text-muted: #a0a0b0;
        --accent: #e94560;
        --border: #2a2a4a;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        background: var(--bg);
        color: var(--text);
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
        line-height: 1.6;
    }}
    h1 {{
        font-size: 1.8rem;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.5rem;
        margin-bottom: 0.3rem;
    }}
    .timestamp {{
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-bottom: 2rem;
    }}
    h2 {{
        font-size: 1.3rem;
        color: var(--accent);
        margin-top: 2rem;
        margin-bottom: 0.8rem;
    }}
    h3 {{ font-size: 1.1rem; margin-top: 1.5rem; margin-bottom: 0.5rem; }}
    .note {{
        background: var(--surface);
        border-left: 3px solid var(--accent);
        padding: 0.6rem 1rem;
        margin: 0.8rem 0;
        border-radius: 0 4px 4px 0;
    }}
    .image-row {{
        display: flex;
        gap: 1rem;
        overflow-x: auto;
        padding: 0.5rem 0;
        margin: 0.8rem 0;
    }}
    .image-row .row-item {{
        flex: 1 1 0;
        min-width: 0;
        text-align: center;
    }}
    .image-row img {{
        width: 100%;
        height: auto;
        border-radius: 4px;
        border: 1px solid var(--border);
    }}
    figure {{
        text-align: center;
        margin: 0.8rem 0;
    }}
    figure img {{
        max-width: 100%;
        border-radius: 4px;
        border: 1px solid var(--border);
    }}
    figcaption {{
        color: var(--text-muted);
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}
    table {{
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.9rem;
    }}
    th, td {{
        border: 1px solid var(--border);
        padding: 0.5rem 0.8rem;
        text-align: center;
    }}
    th {{
        background: var(--surface);
        font-weight: 600;
    }}
    .image-grid img {{
        max-width: 256px;
        border-radius: 4px;
    }}
    .row-label {{
        font-weight: 600;
        text-align: right;
        padding-right: 1rem;
        white-space: nowrap;
    }}
    hr {{
        border: none;
        border-top: 1px solid var(--border);
        margin: 2rem 0;
    }}
</style>
</head>
<body>
<h1>{self.title}</h1>
<p class="timestamp">Generated {timestamp}</p>
{body}
</body>
</html>"""

    def save(self, path: str = "diagnostics.html"):
        """Write the report to an HTML file."""
        html = self._render()
        Path(path).write_text(html)
        print(f"Report saved to {path}")
