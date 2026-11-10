"""Paper Fig. "Primer classification performance of each scoring head".

    python analysis/plot_classification.py

Reads results/encoder/fusion_f1.csv; writes figures/classification.{pdf,svg,png}.
"""
import argparse
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(HERE)
LABEL = "tab:ae9_classification"
METRICS = ["Precision", "Recall", "F$_1$ score", "AUROC"]
# table row key -> (legend label, colour). Okabe-Ito, validated
# (validate_palette.js, light mode); fixed order, so position is a second cue.
SCORES = {
    r"S_A":                 (r"$S_A$ ($\mathcal{M}_{\Delta G}$)",  "#0072B2"),
    r"C_{\mathrm{marg}}":   (r"$C_{\mathrm{marg}}$ ($\mathcal{M}_C$)",   "#E69F00"),
    r"C_{\mathrm{inter}}":  (r"$C_{\mathrm{inter}}$ ($\mathcal{M}_C$)",  "#009E73"),
    r"S_P":                 (r"$S_P$ ($\mathcal{M}_{\Pi}$)",        "#CC79A7"),
    r"S_{\mathrm{comb}}":   (r"$S_{\mathrm{comb}}$ (combined)", "#D55E00"),
}
SHORT = {"Saccharomyces cerevisiae": "S. cerevisiae", "Escherichia coli": "E. coli",
         r"Bacteriophage $\lambda$": "Phage λ"}
ITALIC = {"S. cerevisiae", "E. coli"}
INK, GRID = "#1a1a1a", "#e6e6e6"
MM = 1 / 25.4


def _strip(cell):
    return re.sub(r"\\(?:textbf|underline|textit)\{([^}]*)\}", r"\1", cell).strip()


def load_table(path):
    """-> (corpora, {score_key: (n_corpora, 4) array})."""
    tex = open(path).read()
    if r"\label{%s}" % LABEL not in tex:
        raise SystemExit(f"{LABEL} not found in {path}")
    i = tex.index(r"\label{%s}" % LABEL)
    body = tex[i:tex.index(r"\end{tabularx}", i)]
    corpora, rows, corpus = [], {}, None
    for line in body.splitlines():
        m = re.match(r"\s*\\multirow\{\d+\}\{\*\}\{(.+)\}\s*$", line)
        if m:
            corpus = _strip(m.group(1))
            corpora.append(corpus)
            continue
        m = re.match(r"\s*&\s*\$(.+?)\$\s*&(.+?)\\\\", line)
        if corpus and m:
            vals = [float(_strip(c)) for c in m.group(2).split("&")]
            rows.setdefault(m.group(1).strip(), []).append(vals)
    missing = [k for k in SCORES if k not in rows]
    if missing:
        raise SystemExit("scores not found in the table: %s" % ", ".join(missing))
    return corpora, {k: np.array(v) for k, v in rows.items()}


# fusion_f1.csv equivalents of the table rows.
CSV_COMBO = {r"S_A": "A", r"C_{\mathrm{marg}}": "C_marg", r"C_{\mathrm{inter}}": "C_inter",
             r"S_P": "P", r"S_{\mathrm{comb}}": "max(A+C_marg+C_inter+P)"}
CSV_CORPORA = [("test-BakerYeast", "Saccharomyces cerevisiae"), ("test-EColi", "Escherichia coli"),
               ("test-HIV1", "HIV-1"), ("test-Phage", r"Bacteriophage $\lambda$"),
               ("test-SARS", "SARS-CoV-2")]


def load_csv(path):
    import pandas as pd
    d = pd.read_csv(path).set_index(["eval_set", "combo"])
    cols = ["precision", "recall", "f1", "roc_auc"]
    data = {k: np.array([d.loc[(e, c), cols].values.astype(float) for e, _ in CSV_CORPORA])
            for k, c in CSV_COMBO.items()}
    return [name for _, name in CSV_CORPORA], data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(
        REPO_DIR, "results", "encoder", "fusion_f1.csv"))
    ap.add_argument("--tex", default=None,
                    help="read the numbers from this LaTeX table instead of --csv")
    ap.add_argument("--out", default=os.path.join(REPO_DIR, "figures", "classification"))
    args = ap.parse_args()

    if args.tex:
        corpora, data = load_table(args.tex)
    else:
        corpora, data = load_csv(args.csv)
    labels = [SHORT.get(c, c) for c in corpora]
    print("loaded %d corpora: %s" % (len(corpora), ", ".join(labels)))

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })

    # Drawn at the printed width (single-column \textwidth, 131 mm) so the
    # fonts set above are the sizes that reach the page.
    fig, axes = plt.subplots(2, 2, figsize=(131 * MM, 112 * MM), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.85, bottom=0.12,
                        hspace=0.62, wspace=0.08)
    x = np.arange(len(corpora))
    n = len(SCORES)
    w = 0.8 / n
    for m, (ax, metric) in enumerate(zip(axes.flat, METRICS)):
        for k, (key, (lab, col)) in enumerate(SCORES.items()):
            ax.bar(x + (k - (n - 1) / 2) * w, data[key][:, m], width=w, color=col,
                   edgecolor="white", linewidth=0.4, label=lab, zorder=2)
        ax.set_title(metric, fontweight="bold", pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
        for t in ax.get_xticklabels():
            if t.get_text() in ITALIC:
                t.set_fontstyle("italic")
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(corpora) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.15 if m % 2 == 0 else -0.04, 1.06, "abcd"[m], transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="bottom")
    for ax in axes[:, 0]:
        ax.set_ylabel("Score")

    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 1.0), handlelength=1.0, handleheight=0.8,
               handletextpad=0.4, columnspacing=1.3)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig("%s.%s" % (args.out, ext), dpi=600 if ext == "png" else None)
    print("saved %s.{pdf,svg,png}" % args.out)


if __name__ == "__main__":
    main()
