"""Compose the degeneracy and mismatch plots into one compact figure.

Paper Fig. "Degeneracy and mismatch performance across viral targets". Run after
plot_mismatch_summary_both_modes.py, which writes the two input panels.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "figures"
OUTPUT_DIR = ROOT / "figures"
DEGENERACY = SOURCE_DIR / "degenerate_percentage.png"
DEGENERACY_SOURCE = SOURCE_DIR / "degenerate_percentage_olivia_varvamp.png"
MISMATCH = SOURCE_DIR / "mismatch_summary.png"
MISMATCH_COMPOSITE = SOURCE_DIR / "mismatch_summary_composite.png"
OUTPUT_STEM = OUTPUT_DIR / "degenerate_percentage_mismatch_summary"


def main() -> None:
    if not DEGENERACY.exists():
        Image.open(DEGENERACY_SOURCE).convert("RGB").save(DEGENERACY)

    mismatch = Image.open(MISMATCH_COMPOSITE if MISMATCH_COMPOSITE.exists() else MISMATCH).convert("RGB")
    images = [Image.open(DEGENERACY).convert("RGB"), mismatch]

    figure, axes = plt.subplots(
        1, 2, figsize=(13.5, 5.6), gridspec_kw={"width_ratios": [1, 2.0]}
    )
    for axis, image in zip(axes, images):
        axis.imshow(image)
        axis.axis("off")
    figure.subplots_adjust(0.02, 0.05, 0.995, 0.86, wspace=0.10)
    label_style = {
        "fontsize": 14,
        "fontweight": "bold",
        "va": "bottom",
        "ha": "left",
        "color": "#1a1a1a",
        "clip_on": False,
    }
    # Match the standalone mismatch plot's axis-relative panel labels.
    axes[0].text(-0.06, 1.02, "a", transform=axes[0].transAxes, **label_style)
    axes[1].text(-0.06, 1.02, "b", transform=axes[1].transAxes, **label_style)
    axes[1].text(-0.06, 0.52, "c", transform=axes[1].transAxes, **label_style)
    figure.savefig(f"{OUTPUT_STEM}.png", dpi=600, bbox_inches="tight", pad_inches=0)
    figure.savefig(f"{OUTPUT_STEM}.pdf", dpi=600, bbox_inches="tight", pad_inches=0)
    figure.savefig(f"{OUTPUT_STEM}.svg", dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(figure)


if __name__ == "__main__":
    main()
