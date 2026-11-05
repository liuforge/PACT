"""Appendix table of the hyper-parameters selected on validation for PACT and of the baseline settings.

PACT: for every (dataset, H) the grid configuration with the lowest seed-2024 validation MSE (as in make_tables.load);
baselines: the settings of tools/make_jobs.py (official scripts) and the validation-selected learning rate of the
linear models.

  python tools/appendix_hparams.py --root <results>... --out ../revision/tables/hparams.tex
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_jobs import TIMECMA_OFFICIAL, _IT, _PT_ETT, _TN  # noqa: E402
from make_tables import DATASETS, horizons  # noqa: E402


def selected(roots, prefix):
    best = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "*", f"{prefix}*", "*_s2024", "results.json")):
            r = json.load(open(f))
            c = r["config"]
            if c["seq_len"] not in (96, 36) or c.get("train_percent", 100) != 100:
                continue
            k = (c["dataset"], c["pred_len"])
            if k not in best or r["val"]["mse"] < best[k][0]:
                best[k] = (r["val"]["mse"], c)
    return {k: v[1] for k, v in best.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pact = selected(a.root, "pact__")
    rl = selected(a.root, "rlinear__")
    dl = selected(a.root, "dlinear__")
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Hyper-parameters selected on the validation set (seed 2024). "
             r"PACT: learning rate / width $D$ / BDCA feed-forward layer (1: on); RLinear and DLinear: learning rate. "
             r"All other settings are given in Section~\ref{sec:setup} and the official scripts of the baselines.}",
             r"\label{tab:hparams}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l l c c c}", r"\toprule",
             r"Dataset & $H$ & PACT & RLinear & DLinear \\", r"\midrule"]
    for ds in DATASETS:
        rows = []
        for H in horizons(ds):
            c = pact.get((ds, H))
            if c is None:
                continue
            cell = f"{c['lr']:g} / {c['d_model']} / {c['bdca_ffn']}"
            r = rl.get((ds, H))
            d = dl.get((ds, H))
            rows.append(("" if rows else ds) + f" & {H} & {cell} & " + (f"{r['lr']:g}" if r else "--") + " & "
                        + (f"{d['lr']:g}" if d else "--") + r" \\")
        if rows:
            lines += rows + [r"\midrule"]
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table}", ""]
    # baseline architecture settings (from the official scripts)
    lines += [r"\begin{table}[t]", r"\centering", r"\caption{Architecture settings of the trained baselines, taken from "
              r"their official scripts (TimeCMA: learning rate / channels / encoder layers / decoder layers / dropout / "
              r"batch; iTransformer: $d_{model}$ / $d_{ff}$ / layers; TimesNet: $d_{model}$ / $d_{ff}$). PatchTST uses "
              r"the per-horizon settings of the Time-Series-Library scripts on ETT and 3 layers, 16 heads, $d_{model}=128$ "
              r"elsewhere (4 heads, $d_{model}=16$ on ILI).}", r"\label{tab:baseline_hparams}",
              r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l l c c c}", r"\toprule",
              r"Dataset & $H$ & TimeCMA & iTransformer & TimesNet \\", r"\midrule"]
    for ds in DATASETS:
        if ds not in TIMECMA_OFFICIAL:
            continue
        rows = []
        for H in horizons(ds):
            t = TIMECMA_OFFICIAL[ds][H]
            it = _IT.get(ds)
            tn = _TN.get(ds)
            rows.append(("" if rows else ds) + f" & {H} & " + " / ".join(f"{x:g}" for x in t) + " & "
                        + (" / ".join(str(x) for x in it) if it else "--") + " & "
                        + (" / ".join(str(x) for x in tn) if tn else "--") + r" \\")
        lines += rows + [r"\midrule"]
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table}"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"written {a.out}: {len(pact)} PACT cells")


if __name__ == "__main__":
    main()
