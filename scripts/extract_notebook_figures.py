"""Extract embedded PNG figures from notebook cell outputs into docs/figures/.

The .ipynb files store rendered plots as base64-encoded PNG strings inside
cell outputs. This script decodes those without re-executing the notebooks.
"""

import base64
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "docs" / "figures"

EXTRACTIONS = [
    ("notebooks/01_data_exploration.ipynb", 5, "class_distribution.png"),
    ("notebooks/02_baseline_model.ipynb", 12, "baseline_confusion_matrices.png"),
    ("notebooks/03_cnn_model.ipynb", 13, "cnn_training_curves.png"),
    ("notebooks/03_cnn_model.ipynb", 16, "cnn_confusion_matrix.png"),
    ("notebooks/04_cafe_scoring.ipynb", 13, "cafe_scores_bar.png"),
    ("notebooks/04_cafe_scoring.ipynb", 14, "class_distribution_per_cafe.png"),
]


def find_png_in_cell(cell):
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "image/png" in data:
            b64 = data["image/png"]
            if isinstance(b64, list):
                b64 = "".join(b64)
            return base64.b64decode(b64.replace("\n", ""))
    return None


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for nb_rel, cell_idx, filename in EXTRACTIONS:
        nb_path = ROOT / nb_rel
        with open(nb_path) as f:
            nb = json.load(f)
        cell = nb["cells"][cell_idx]
        png_bytes = find_png_in_cell(cell)
        if png_bytes is None:
            print(f"SKIP  {nb_rel} cell {cell_idx} — no PNG output")
            continue
        out_path = FIGURES_DIR / filename
        out_path.write_bytes(png_bytes)
        print(f"WROTE {out_path.relative_to(ROOT)} ({len(png_bytes) // 1024} KB)")


if __name__ == "__main__":
    main()
