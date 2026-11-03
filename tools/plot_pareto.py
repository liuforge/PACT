"""Accuracy-cost trade-off: relative test MSE against parameters and training time.

Relative MSE: for every dataset, each model's horizon-averaged test MSE is divided by the median over the trained
models on that dataset; the ratios are then averaged over the datasets on which every plotted model has results.
Parameters and s/epoch are taken from the ETTm1, H=96 runs; for the paper, use --cost from
tools/benchmark_efficiency.py, which times every model on the same idle GPU.

  python tools/plot_pareto.py --root <results>... --out figures/pareto.pdf
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import horizons, load, mean  # noqa: E402

PACT, TRAINED, FM = "#2a78d6", "#8a8984", "#eb6834"  # slot 1, neutral, slot 2
INK, MUTED = "#0b0b0b", "#52514e"
OFFSET = {"PACT": (6, -11), "PatchTST": (6, -10), "iTransformer": (-8, 7), "RLinear": (6, 4), "DLinear": (6, 4),
          "TimeCMA": (6, 4), "TimesNet": (6, 4)}
MODELS = [("pact", "PACT", PACT), ("timecma", "TimeCMA", TRAINED), ("patchtst", "PatchTST", TRAINED),
          ("itransformer", "iTransformer", TRAINED), ("timesnet", "TimesNet", TRAINED), ("dlinear", "DLinear", TRAINED),
          ("rlinear", "RLinear", TRAINED), ("zs_timesfm-3.0-pytorch_ctxL", "TimesFM-3", FM),
          ("zs_Toto-2.0-313m_ctxL", "Toto-2 313M", FM), ("zs_Toto-2.0-22m_ctxL", "Toto-2 22M", FM),
          ("zs_TimeMoE-200M_ctxL", "Time-MoE 200M", FM)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2,Weather")
    p.add_argument("--exclude", default="", help="models to leave out (e.g. diverging ones)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    R = {}
    for root in a.root:
        for k, v in load(root).items():
            seen = {r["seed"] for r in R.get(k, [])}
            R.setdefault(k, []).extend(r for r in v if r["seed"] not in seen)
    models = [m for m in MODELS if m[0] not in a.exclude.split(",")]
    dss = [d for d in a.datasets.split(",")
           if all(all((d, m, h) in R for h in horizons(d)) for m, _, _ in models)]
    if not dss:
        raise SystemExit("no dataset has results for every model")
    avg = {(d, m): np.mean([mean(R[(d, m, h)], "mse") for h in horizons(d)]) for d in dss for m, _, _ in models}
    rel = {}
    for m, _, _ in models:
        rel[m] = np.mean([avg[(d, m)] / np.median([avg[(d, t)] for t, _, c in models if c != FM]) for d in dss])
    top = max(rel[m] for m, _, c in models if c != FM) * 1.06
    bottom = min(rel.values()) * 0.97
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, key, xlabel in ((axes[0], "params", "trainable parameters (M, log scale)"),
                            (axes[1], "epoch_s", "training time (s / epoch, log scale)")):
        n_off = 0
        for m, name, color in models:
            runs = R.get(("ETTm1", m, 96))
            if not runs:
                continue
            x = runs[0][key] / 1e6 if key == "params" else mean(runs, key)
            if not x:  # zero-shot models have no training time
                continue
            if rel[m] > top:  # off-scale: marker on the top edge with its value
                ax.scatter([x], [top], s=46, marker="^", color=color, edgecolor="white", linewidth=1.5, zorder=3,
                           clip_on=False)
                ax.annotate(f"{name} ({rel[m]:.2f})", (x, top), xytext=(-4, -12 - 10 * n_off),
                            textcoords="offset points", fontsize=7, color=INK, ha="right")
                n_off += 1
                continue
            ax.scatter([x], [rel[m]], s=46, color=color, edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(name, (x, rel[m]), xytext=OFFSET.get(name, (5, 3)), textcoords="offset points", fontsize=7,
                        color=INK if color != TRAINED else MUTED)
        ax.set_xscale("log")
        ax.set_ylim(bottom, top)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.axhline(1.0, color="#e4e3df", lw=1, zorder=0)
        ax.set_xlabel(xlabel, color=MUTED)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("relative test MSE (lower is better)", color=MUTED)
    fig.suptitle(f"Accuracy vs. cost (relative MSE over {', '.join(dss)})", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print("datasets:", dss, "| relative MSE:", {n: round(rel[m], 3) for m, n, _ in models})


if __name__ == "__main__":
    main()
