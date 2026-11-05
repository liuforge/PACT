"""Efficiency figure (replaces the efficiency table in the paper; the full table is in the supplementary material).

(a) parameters against inference latency, marker area proportional to peak GPU memory; (b) training step time of
the trained models. All numbers are measured by benchmark_efficiency.py on one idle GPU (ETTm1, H=96, batch 32).

  python tools/plot_efficiency.py --bench <efficiency.json> --out ../revision/figures/efficiency.pdf
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palette import C1 as FM, C2 as TRAINED, C3 as PACT, GRID, INK, MUTED, apply_style  # noqa: E402
OFFSET = {  # label placement (points) chosen so that labels do not collide
    "PACT": (6, 4), "TimeCMA": (6, -2), "PatchTST": (6, -6), "iTransformer": (6, -4), "TimesNet": (-7, 3),
    "DLinear": (6, 0), "RLinear": (-4, 8), "TimesFM-3": (-8, 6), "Toto-2 22M": (6, -2), "Toto-2 313M": (-8, 6),
    "Time-MoE 50M": (6, -6), "Time-MoE 200M": (-8, 6)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rows = [r for r in json.load(open(a.bench))["results"]
            if "per variate" not in r["model"] and "ctx 512" not in r["model"] and r["model"] != "PACT (no anchor)"]
    apply_style(plt, 7)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.45), gridspec_kw={"width_ratios": [1.45, 1]})
    for r in rows:
        name = r["model"].replace(" (ctx 96)", "")
        zs = r.get("s_per_step") is None
        col = PACT if name == "PACT" else (FM if zs else TRAINED)
        ax.scatter(r["params"] / 1e6, r["infer_ms_per_batch"], s=8 + r["peak_mem_mib"] / 12, color=col, alpha=0.85,
                   edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy = OFFSET.get(name, (6, 0))
        ax.annotate(name, (r["params"] / 1e6, r["infer_ms_per_batch"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6, color=INK if name == "PACT" else MUTED, ha="right" if dx < 0 else "left", va="center",
                    fontweight="bold" if name == "PACT" else "normal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("trainable / total parameters (M, log)", color=MUTED, fontsize=6.5)
    ax.set_ylabel("inference (ms per batch, log)", color=MUTED, fontsize=6.5)
    ax.set_title("(a) Size and inference cost (area: peak memory)", loc="left", fontsize=7.5, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.5, zorder=0)
    for c, lab in [(PACT, "PACT"), (TRAINED, "trained baselines"), (FM, "foundation models (zero-shot)")]:
        ax.scatter([], [], s=22, color=c, label=lab)
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.1, borderaxespad=0.1)
    ax.set_xlim(0.004, 3000)
    ax.set_ylim(0.15, 4000)
    tr = sorted([r for r in rows if r.get("s_per_step")], key=lambda r: r["s_per_step"])
    names = [r["model"] for r in tr]
    vals = [r["s_per_step"] * 1000 for r in tr]
    bx.barh(range(len(tr)), vals, height=0.62, color=[PACT if n == "PACT" else TRAINED for n in names], zorder=3)
    for i, (n, v, r) in enumerate(zip(names, vals, tr)):
        bx.text(v + 0.6, i, f"{v:.1f} ms  ·  {r['peak_mem_mib']:.0f} MiB", va="center", fontsize=6,
                color=INK if n == "PACT" else MUTED)
    bx.set_yticks(range(len(tr)))
    bx.set_yticklabels(names, fontsize=6.5)
    bx.set_xlim(0, max(vals) * 1.6)
    bx.set_xlabel("training step (ms)  ·  peak memory", color=MUTED, fontsize=6.5)
    bx.set_title("(b) Training cost of the trained models", loc="left", fontsize=7.5, color=INK)
    bx.tick_params(axis="y", length=0)
    bx.grid(True, axis="x", color=GRID, lw=0.5, zorder=0)
    for a_ in (ax, bx):
        a_.tick_params(colors=MUTED, labelsize=6.5)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    bx.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
