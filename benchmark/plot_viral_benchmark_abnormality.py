"""Paper Fig. "Abnormality rates of primers designed by each tool".

    python plot_viral_benchmark_abnormality.py [--heads]

Reads output/tables_viral_benchmark.tex; writes
figures/viral_benchmark_abnormality_overall.
"""
import argparse
import os
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ---------------- data (parsed from tables_viral_benchmark.tex) ----------------
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("tex", nargs="?", default=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "tables_viral_benchmark.tex"),
    help=".tex holding the two tables [output/tables_viral_benchmark.tex]")
_ap.add_argument("--heads", action="store_true",
                 help="also draw the per-head heatmap figure")
_ap.add_argument("--heads-only", action="store_true",
                 help="draw only the per-head heatmap figure")
ARGS = _ap.parse_args()
if ARGS.heads_only:
    ARGS.heads = True
TEX = ARGS.tex
TABLE_LABELS = ["tab:viral_benchmark_evaluation", "tab:viral_benchmark_evaluation_cont"]
TOOLS = ["Olivia-NUPACK", "Olivia-deg", "Olivar",
         "PrimalScheme3", "varVAMP"]
HEADS = ["Head A", "Head C (marginal)", "Head C (interrelated)", "Head P"]
# short labels for the axes (define in the figure legend)
ABBREV = {
    "SARS-CoV-2": "SARS-CoV-2", "Hepatitis E": "HEV", "Hepatitis C": "HCV",
    "HIV-1": "HIV-1", "Measles": "MeV", "Rabies": "RABV", "BoDV-1": "BoDV-1",
    "Poliovirus 1": "PV-1", "Rat Hepatitis E": "Rat HEV", "Hepatitis A": "HAV",
    "RSV-A": "RSV-A", "RSV-B": "RSV-B",
}


def _strip(cell):
    return re.sub(r"\\(?:textbf|underline|textit)\{([^}]*)\}", r"\1", cell).strip()


def load_tables(path, labels):
    """Return ordered {virus: [[A, Cm, Ci, P, Overall] per tool in TOOLS]}."""
    tex = open(path).read()
    data = {}
    for label in labels:
        if r"\label{%s}" % label not in tex:
            raise SystemExit(f"{label} not found in {path}; pass the .tex that holds "
                             "the viral benchmark tables as the first argument")
        i = tex.index(r"\label{%s}" % label)
        body = tex[i:tex.index(r"\end{tabularx}", i)]
        virus = None
        for line in body.splitlines():
            line = line.split("%")[0].strip() if not line.lstrip().startswith("%") else ""
            m = re.match(r"\\multirow\{\d+\}\{\*\}\{\\textbf\{(.+?)\}\}", line)
            if m:
                virus = m.group(1)
                data[virus] = {}
                continue
            if virus and line.startswith("&"):
                cells = [_strip(c) for c in line.rstrip("\\ ").split("&")[1:]]
                data[virus][cells[0]] = [float(c) for c in cells[1:6]]
    return {v: [rows[t] for t in TOOLS] for v, rows in data.items()}


DATA = load_tables(TEX, TABLE_LABELS)
VIRUSES = list(DATA)
arr = np.array([DATA[v] for v in VIRUSES])  # (virus, tool, metric)
print(f"loaded {len(VIRUSES)} viruses:", ", ".join(VIRUSES))

# Shared categorical palette with the mismatch-summary figure.
COLORS = {
    "Olivia-NUPACK": "#56B4E9",
    "Olivia-deg": "#D55E00",
    "Olivar": "#0072B2",
    "PrimalScheme3": "#E69F00",
    "varVAMP": "#009E73",
}
DISPLAY_NAMES = {"Olivia-NUPACK": "Olivia-nondeg"}
INK, INK2, GRID = "#1a1a1a", "#555555", "#e3e3e3"

# ---------------- Nature-style rcParams ----------------
MM = 1 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(out, exist_ok=True)


def save(fig, stem):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"),
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    print("saved", os.path.join(out, stem + ".{pdf,svg,png}"))


# ---------------- Figure 1: overall abnormality, grouped bars ----------------
if not ARGS.heads_only:
    # Drawn at the printed width (\textwidth, 131 mm) so these font sizes
    # are the ones on the page.
    fig = plt.figure(figsize=(131 * MM, 68 * MM))
    ax = fig.add_axes([0.085, 0.2, 0.905, 0.7])
    groups = [ABBREV.get(v, v) for v in VIRUSES] + ["Mean"]
    overall = arr[:, :, 4]                                    # (virus, tool)
    mean = overall.mean(axis=0)
    n_t = len(TOOLS)
    width = 0.16
    x = np.arange(len(groups), dtype=float)
    x[-1] += 0.40                                             # separate the mean group

    for i, tool in enumerate(TOOLS):
        vals = np.append(overall[:, i], mean[i])
        pos = x + (i - (n_t - 1) / 2) * width
        ax.bar(pos, vals, width=width * 0.86, color=COLORS[tool],
               label=DISPLAY_NAMES.get(tool, tool),
               edgecolor="none", zorder=2)
        # per-virus points on the mean bar
        jitter = np.linspace(-width * 0.22, width * 0.22, len(VIRUSES))
        ax.scatter(pos[-1] + jitter, overall[:, i], s=3, facecolor="white",
                   edgecolor=INK, linewidth=0.35, zorder=4)

    # mark best tool per group
    for g in range(len(groups)):
        vals = overall[g] if g < len(VIRUSES) else mean
        b = int(np.argmin(vals))
        pos = x[g] + (b - (n_t - 1) / 2) * width
        top = vals[b] if g < len(VIRUSES) else overall[:, b].max()
        ax.text(pos, top + 0.6, "*", ha="center", va="bottom", fontsize=8, color=INK)

    ax.axvline((x[-2] + x[-1]) / 2, color=INK2, lw=0.4, ls=(0, (2, 2)), zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups[:-1] + ["Mean"], rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_ylabel("Overall abnormality rate (%)")
    # Y limit from the data, rounded up to the next 10.
    top = float(np.nanmax(overall)) if np.isfinite(overall).any() else 60.0
    ymax = max(60.0, 10.0 * np.ceil(top * 1.05 / 10.0))
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 0.01, 10))
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.legend(ncol=5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.0, handleheight=0.8, columnspacing=1.0,
              handletextpad=0.4, borderaxespad=0.2)
    save(fig, "viral_benchmark_abnormality_overall")

# ---------------- Figure 2: per-head grouped bars ----------------
if ARGS.heads:
    # One row per head; viruses on x, the five tools as bars.
    fig, axes = plt.subplots(len(HEADS), 1, figsize=(180 * MM, 170 * MM),
                             sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.94, bottom=0.05,
                        hspace=0.32)
    n_t = len(TOOLS)
    width = 0.16
    x = np.arange(len(VIRUSES), dtype=float)
    for h, (axh, head) in enumerate(zip(axes, HEADS)):
        M = arr[:, :, h]                                   # (virus, tool)
        for i, tool in enumerate(TOOLS):
            axh.bar(x + (i - (n_t - 1) / 2) * width, M[:, i], width=width * 0.86,
                    color=COLORS[tool], label=DISPLAY_NAMES.get(tool, tool),
                    edgecolor="none", zorder=2)
        top = M.max()
        step = 2 if top <= 12 else 5 if top <= 25 else 10
        ymax = step * np.ceil(top * 1.08 / step)
        axh.set_ylim(0, ymax)
        axh.set_yticks(np.arange(0, ymax + 0.01, step))
        axh.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        axh.yaxis.grid(True, color=GRID, lw=0.4, zorder=0)
        axh.set_axisbelow(True)
        for sp in ("top", "right"):
            axh.spines[sp].set_visible(False)
        axh.tick_params(axis="x", length=0)
        axh.set_ylabel("Abnormality rate (%)")
        axh.set_title(head, loc="left", pad=3, fontweight="bold")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([ABBREV.get(v, v) for v in VIRUSES])
    axes[0].legend(ncol=n_t, frameon=False, loc="lower center",
                   bbox_to_anchor=(0.5, 1.12), handlelength=1.0, handleheight=0.8,
                   columnspacing=1.6, borderaxespad=0.2)
    save(fig, "viral_benchmark_abnormality_heads")
