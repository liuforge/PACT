"""Generate the LaTeX tables of the revised manuscript from results.json files.

Every number in the paper's result tables is produced here; nothing is typed by hand.

  python tools/make_tables.py --root <results> --out ../revision/tables
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Solar", "PEMS08", "ECL", "Traffic", "ILI"]
# runs with identical configurations under two tags: the full ablation model and the GPT-2 control are the
# selected main-table configuration, so missing (dataset, H, seed) cells fall back to the main-table run
# datasets whose series are very likely in Sundial's pre-training corpus (TimeBench includes the Chronos corpus,
# with the NREL solar data of 2006, and LOTSA, with PEMS08 and Traffic); Sundial itself excludes ETT, ECL and Weather
SUNDIAL_SEEN = ("Solar", "PEMS08", "Traffic")
ALIASES = {"abl_full": "pact", "llm_gpt2": "pact"}
HORIZONS = {"ILI": [24, 36, 48, 60], "PEMS08": [12, 24, 48, 96]}
SHADE_NOTE = r" Shading: \swatch{upc!30}{higher} / \swatch{downc!30}{lower} error than the reference; bold: $p<0.05$."
MAIN_MODELS = [  # (tag after selection, column title)
    ("pact", "PACT"), ("timecma", "TimeCMA"), ("patchtst", "PatchTST"), ("itransformer", "iTrans."),
    ("timesnet", "TimesNet"), ("dlinear", "DLinear"), ("rlinear", "RLinear"),
    ("zs_timesfm-3.0-pytorch-clip1_ctxL", "TimesFM-3$^\\ast$"), ("zs_Toto-2.0-313m_ctxL", "Toto-2"),
    ("zs_TimeMoE-200M_ctxL", "Time-MoE"),
]


def horizons(ds):
    return HORIZONS.get(ds, [96, 192, 336, 720])


def load(root, select_seed=2024):
    """runs[(dataset, tag, H)] = list of dicts; '<name>__<hp>' variants collapsed by seed-2024 validation MSE."""
    raw = collections.defaultdict(list)
    for f in glob.glob(os.path.join(root, "*", "*", "*", "results.json")):
        r = json.load(open(f))
        c = r["config"]
        tag = c["tag"]
        if tag.startswith("zs_") and tag.endswith(f"_ctx{c['seq_len']}"):  # same context as the protocol look-back
            tag = tag[: tag.rindex("_ctx")] + "_ctxL"
        raw[(c["dataset"], tag, c["pred_len"])].append(dict(
            seed=c["seed"], mse=r["test"]["mse"], mae=r["test"]["mae"], val=r["val"]["mse"],
            params=r.get("n_params"), epoch_s=r.get("train_s_per_epoch"), mem=r.get("peak_mem_mib"),
            infer=r["test"].get("infer_s_per_batch"), epochs=r.get("epochs_run"), transfer=r.get("transfer")))
    best = {}
    for (ds, tag, H), runs in raw.items():
        if "__" not in tag:
            continue
        base = tag.split("__")[0]
        sel = [x["val"] for x in runs if x["seed"] == select_seed] or [x["val"] for x in runs]
        v = float(np.mean(sel))
        if (ds, base, H) not in best or v < best[(ds, base, H)][0]:
            best[(ds, base, H)] = (v, tag)
    out = {k: v for k, v in raw.items() if "__" not in k[1]}
    for (ds, base, H), (_, tag) in best.items():
        out[(ds, base, H)] = raw[(ds, tag, H)]
    return out


def mean(runs, key):
    return float(np.mean([r[key] for r in runs]))


def shade(v, cap=8.0, top=40):
    """Cell background for a relative change v (in %): red when the error increases, blue when it decreases,
    intensity proportional to |v| and saturating at cap."""
    k = int(round(min(abs(v) / cap, 1.0) * top))
    return "" if k < 3 else rf"\cellcolor{{{'upc' if v > 0 else 'downc'}!{k}}}"


def pct_cell(v, p=None):
    v = 0.0 if round(v, 1) == 0 else v  # no "-0.0"
    return shade(v) + f"${v:+.1f}$\\%"


def p_txt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def p_cell(p):
    return rf"\sig{{{p_txt(p)}}}" if p < 0.05 else p_txt(p)


def fmt_rank(vals, digits=3):
    """Best in red and bold, second in blue and underlined (ties share the rank)."""
    finite = sorted({round(v, digits) for v in vals if v is not None})
    out = []
    for v in vals:
        if v is None:
            out.append("--")
            continue
        s = f"{v:.{digits}f}"
        r = round(v, digits)
        out.append(r"\first{" + s + "}" if r == finite[0] else (r"\runnerup{" + s + "}" if len(finite) > 1 and r == finite[1] else s))
    return out


def main_table(R, models, caption, label, datasets=None):
    cols = "c c " + r">{\columncolor{oursbg}}c>{\columncolor{oursbg}}c " + " ".join(["cc"] * (len(models) - 1))
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\setlength{\tabcolsep}{3.2pt}", r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.86\textheight}",
             r"\begin{tabular}{" + cols + "}", r"\toprule",
             "& & " + " & ".join(r"\multicolumn{2}{c}{" + t + "}" for _, t in models) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(models))),
             "Dataset & $H$ & " + " & ".join(["MSE & MAE"] * len(models)) + r" \\", r"\midrule"]
    wins, wins_mae = collections.Counter(), collections.Counter()
    gaps, gaps_mae = gap_to_best(R, models), gap_to_best(R, models, key="mae")
    for ds in datasets or DATASETS:
        rows = []
        for H in horizons(ds) + ["Avg"]:
            mse, mae = [], []
            for tag, _ in models:
                if H == "Avg":
                    ok = all((ds, tag, h) in R for h in horizons(ds))
                    mse.append(np.mean([mean(R[(ds, tag, h)], "mse") for h in horizons(ds)]) if ok else None)
                    mae.append(np.mean([mean(R[(ds, tag, h)], "mae") for h in horizons(ds)]) if ok else None)
                else:
                    runs = R.get((ds, tag, H))
                    mse.append(mean(runs, "mse") if runs else None)
                    mae.append(mean(runs, "mae") if runs else None)
            if all(v is None for v in mse):
                continue
            # a zero-shot model whose error explodes on some windows (non-finite, or more than ten times the median
            # of the trained models in this cell) is reported as diverged and not ranked
            ref = [v for (tag, _), v in zip(models, mse) if v is not None and not tag.startswith("zs_") and np.isfinite(v)]
            fin = [v for v in mse if v is not None and np.isfinite(v)]
            lim = 10 * float(np.median(ref)) if ref else (10 * min(fin) if fin else np.inf)
            div = [v is not None and (not np.isfinite(v) or v > lim) for v in mse]
            mse = [None if d else v for v, d in zip(mse, div)]
            mae = [None if d else v for v, d in zip(mae, div)]
            if H != "Avg":
                best = min(v for v in mse if v is not None)
                for (tag, _), v in zip(models, mse):
                    if v is not None and round(v, 3) == round(best, 3):
                        wins[tag] += 1
                best_mae = min(v for v in mae if v is not None)
                for (tag, _), v in zip(models, mae):
                    if v is not None and round(v, 3) == round(best_mae, 3):
                        wins_mae[tag] += 1
            cells = [r"\multicolumn{2}{c}{div.}" if d else f"{a} & {b}" for a, b, d in zip(fmt_rank(mse), fmt_rank(mae), div)]
            if H == "Avg":  # dataset name, rotated, spanning the rows of the dataset (drawn last, above the shading)
                n = len(rows) + 1
                rows.append(r"\rowcolor{avgbg}\cellcolor{white}\multirow{-" + str(n) + r"}{*}{\rotatebox[origin=c]{90}{"
                            + ds + "}}" + " & Avg & " + " & ".join(cells) + r" \\")
            else:
                rows.append(f" & {H} & " + " & ".join(cells) + r" \\")
        if rows:
            lines += rows + [r"\midrule"]
    lines[-1] = r"\midrule"
    # one summary row per metric: first places and median gap to the best model of each cell
    for name, w, g in (("MSE", wins, gaps), ("MAE", wins_mae, gaps_mae)):
        top_w, top_g = max(w.values(), default=0), min(g.values(), default=0)
        fw = lambda t: (r"\first{%d}" if w[t] == top_w and top_w else "%d") % w[t]  # noqa: E731
        fg = lambda t: (r"\first{%+.1f\%%}" if g[t] == top_g else r"%+.1f\%%") % g[t]  # noqa: E731
        lines.append(rf"\multicolumn{{2}}{{l}}{{1st count ({name})}} & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{fw(t)}}}" for t, _ in models) + r" \\")
        lines.append(rf"\multicolumn{{2}}{{l}}{{Median gap to best ({name})}} & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{fg(t)}}}" if t in g else r"\multicolumn{2}{c}{--}" for t, _ in models) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
    return "\n".join(lines)


def gap_to_best(R, models, key="mse"):
    """Median relative excess of test MSE over the best model of each (dataset, horizon) cell. Cells are those where
    every trained model has results; the best is taken over all models present in the cell, and a zero-shot model
    is averaged over the cells it has."""
    trained = [t for t, _ in models if not t.startswith("zs_")]
    cells = [(ds, H) for ds in DATASETS for H in horizons(ds) if all((ds, t, H) in R for t in trained)]
    out = {}
    for t, _ in models:
        g = [mean(R[(ds, t, H)], key) / min(mean(R[(ds, u, H)], key) for u, _ in models if (ds, u, H) in R) - 1
             for ds, H in cells if (ds, t, H) in R]
        if g:
            out[t] = 100 * np.median(g)  # median: one diverging zero-shot cell must not dominate
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def versus_table(R, ours, baselines, caption, label):
    """Relative test-MSE difference of ours vs. each baseline per dataset (negative = ours better), paired over
    (horizon, seed); Wilcoxon p in parentheses when a dataset has at least six pairs."""
    dss = [ds for ds in DATASETS if any((ds, ours, H) in R for H in horizons(ds))]
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l " + "c " * (len(dss) + 1) + "}", r"\toprule",
             "vs. & " + " & ".join(dss) + r" & All \\", r"\midrule"]
    for tag, name in baselines:
        row, allx, ally = [], [], []
        for ds in dss:
            x, y = [], []
            for H in horizons(ds):
                b = {r["seed"]: r["mse"] for r in R.get((ds, tag, H), [])}
                o = R.get((ds, ours, H), [])
                if tag.startswith("zs_") and b and o:  # deterministic zero-shot: one pair per horizon (seed mean)
                    x.append(np.mean(list(b.values())))
                    y.append(mean(o, "mse"))
                    continue
                for r in o:
                    if r["seed"] in b:
                        x.append(b[r["seed"]])
                        y.append(r["mse"])
            if not x:
                row.append("--")
                continue
            x, y = np.array(x), np.array(y)
            allx += list(x)
            ally += list(y)
            cell = pct_cell(np.mean((y - x) / x) * 100)
            if len(x) >= 6:
                cell += f" ({p_txt(wilcoxon(x, y).pvalue)})"
            row.append(cell)
        x, y = np.array(allx), np.array(ally)
        row.append(pct_cell(np.mean((y - x) / x) * 100) + f" ({p_txt(wilcoxon(x, y).pvalue)})" if len(x) >= 6 else "--")
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def val_test_table(roots, caption, label):
    """How well does validation MSE rank test MSE? Spearman correlation over every trained run (all models,
    configurations and seeds; full training data, L = 96) of each (dataset, horizon)."""
    pts = collections.defaultdict(list)
    for root in roots:
        for f in glob.glob(os.path.join(root, "*", "*", "*", "results.json")):
            r = json.load(open(f))
            c = r["config"]
            if c["tag"].startswith(("zs_", "fs", "tr_", "L512")) or c.get("train_percent", 100) != 100:
                continue
            if c["seq_len"] != (36 if c["dataset"] == "ILI" else 96) or c.get("llm") == "stored" and "raw" in c["tag"]:
                continue
            key = (c["dataset"], c["pred_len"], c["tag"], c["seed"])
            pts[key[:2]].append((key, r["val"]["mse"], r["test"]["mse"]))
    dss = [ds for ds in DATASETS if any(k[0] == ds for k in pts)]
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l " + "c " * 5 + "}", r"\toprule", r"Dataset & \multicolumn{4}{c}{Spearman $\rho$ per horizon} & runs \\", r"\midrule"]
    for ds in dss:
        cells, n = [], 0
        for H in horizons(ds):
            u = {k: (v, t) for k, v, t in pts.get((ds, H), [])}  # one point per (tag, seed)
            if len(u) < 6:
                cells.append("--")
                continue
            v = np.array(list(u.values()))
            cells.append(f"{spearman(v[:, 0], v[:, 1]):+.2f}")
            n += len(u)
        lines.append(f"{ds} & " + " & ".join(cells) + f" & {n}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def full_table(R, models, caption, label):
    """Appendix: mean +- std of MSE for every cell."""
    lines = [r"\begin{table*}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l l " + "c " * len(models) + "}", r"\toprule",
             "Dataset & $H$ & " + " & ".join(t for _, t in models) + r" \\", r"\midrule"]
    for ds in DATASETS:
        first = True
        for H in horizons(ds):
            cells = []
            for tag, _ in models:
                runs = R.get((ds, tag, H))
                if not runs:
                    cells.append("--")
                    continue
                v = np.array([r["mse"] for r in runs])
                cells.append(f"{v.mean():.3f}" + (rf"$\pm${v.std(ddof=1):.3f}" if len(v) > 1 else ""))
            if all(c == "--" for c in cells):
                continue
            lines.append(("" if not first else ds) + f" & {H} & " + " & ".join(cells) + r" \\")
            first = False
        if not first:
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def paired_table(R, ref, variants, caption, label, datasets=None, flagged=None):
    """Relative test-MSE change of each variant vs. ref over (dataset, H, seed) cells, with Wilcoxon p.
    flagged: {tag: datasets} cells shown with a dagger and left out of the All column and p (e.g. possible
    overlap between the test data and the pre-training corpus of the anchor model)."""
    flagged = flagged or {}
    cells = collections.defaultdict(dict)
    for (ds, tag, H), runs in R.items():
        if datasets and ds not in datasets:
            continue
        for r in runs:
            cells[(ds, H, r["seed"])][tag] = r["mse"]
    for c in cells.values():
        for tag, alias in ALIASES.items():
            if tag not in c and alias in c:
                c[tag] = c[alias]
    dss = sorted({k[0] for k in cells if ref in cells[k] and any(t in cells[k] for t, _ in variants)},
                 key=DATASETS.index)
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + SHADE_NOTE + "}", r"\label{" + label + "}",
             r"\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.1}",
             r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{l " + "c" * len(dss) + " | c c}", r"\toprule",
             "Variant & " + " & ".join(dss) + r" & All & $p$ \\", r"\midrule"]
    for tag, name in variants:
        row = []
        allx, ally = [], []
        for ds in dss:
            ks = [k for k in cells if k[0] == ds and ref in cells[k] and tag in cells[k]]
            if len(ks) < 6:  # not run, or still running (fewer than two horizons x three seeds)
                row.append("--")
                continue
            x = np.array([cells[k][ref] for k in ks])
            y = np.array([cells[k][tag] for k in ks])
            if ds in flagged.get(tag, ()):
                row.append(pct_cell(np.mean((y - x) / x) * 100) + r"$^\dagger$")
                continue
            allx += list(x)
            ally += list(y)
            row.append(pct_cell(np.mean((y - x) / x) * 100))
        if len(allx) >= 6:
            x, y = np.array(allx), np.array(ally)
            p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            row += [pct_cell(np.mean((y - x) / x) * 100), p_cell(p)]
        else:
            row += ["--", "--"]
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(lines)


def efficiency_table(R, models, ds, H, caption, label):
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c c}", r"\toprule",
             r"Model & Params (M) & Train (s/epoch) & Inference (ms/batch) & Peak memory (MiB) \\", r"\midrule"]
    for tag, name in models:
        runs = R.get((ds, tag, H))
        if not runs:
            continue
        p = runs[0]["params"]
        ep = mean(runs, "epoch_s")
        inf = np.mean([r["infer"] for r in runs if r["infer"] is not None]) * 1000
        mem = np.mean([r["mem"] for r in runs if r["mem"] is not None]) if any(r["mem"] for r in runs) else None
        lines.append(f"{name} & {p / 1e6:.2f} & " + (f"{ep:.1f}" if ep else "zero-shot") + f" & {inf:.1f} & "
                     + (f"{mem:.0f}" if mem else "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def bench_table(path, caption, label):
    """Efficiency measured by benchmark_efficiency.py on one idle GPU (ETTm1, H=96, batch 32)."""
    b = json.load(open(path))
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + f" GPU: {b['device']}." + "}",
             r"\label{" + label + "}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l c c c c}", r"\toprule",
             r"Model & Params (M) & Train step (ms) & Inference (ms/batch) & Peak memory (MiB) \\", r"\midrule"]
    for r in b["results"]:
        if "per variate" in r["model"]:  # the main results use the joint (multivariate) mode of TimesFM-3
            continue
        step = f"{r['s_per_step'] * 1000:.1f}" if r.get("s_per_step") else "zero-shot"
        params = f"{r['params'] / 1e6:.2f}" if r.get("params") else "--"
        lines.append(f"{r['model']} & {params} & {step} & {r['infer_ms_per_batch']:.1f} & {r['peak_mem_mib']:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def fewshot_table(R, caption, label):
    """Test MSE averaged over datasets, horizons and seeds for each training fraction and anchor condition."""
    conds = [("gpt2", "GPT-2 anchor"), ("learned", "learned vectors"), ("none", "no anchor")]
    pcts = sorted({int(t[2:].split("_")[0]) for (_, t, _) in R if t.startswith("fs") and t[2:].split("_")[0].isdigit()})
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l " + "c " * len(pcts) + "}", r"\toprule",
             "Anchor & " + " & ".join(f"{q}\\%" for q in pcts) + r" \\", r"\midrule"]
    for c, name in conds:
        row = []
        for q in pcts:
            v = [r["mse"] for (ds, t, H), runs in R.items() if t == f"fs{q}_{c}" for r in runs]
            row.append(f"{np.mean(v):.4f}" if v else "--")
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
    pv = []
    for q in pcts:
        cells = collections.defaultdict(dict)
        for (ds, t, H), runs in R.items():
            for c, _ in conds:
                if t == f"fs{q}_{c}":
                    for r in runs:
                        cells[(ds, H, r["seed"])][c] = r["mse"]
        ks = [k for k in cells if "gpt2" in cells[k] and "none" in cells[k]]
        x = np.array([cells[k]["gpt2"] for k in ks]); y = np.array([cells[k]["none"] for k in ks])
        pv.append(p_txt(wilcoxon(x, y).pvalue) if len(ks) >= 6 else "--")
    lines += [r"\midrule", r"$p$ (GPT-2 vs.\ none) & " + " & ".join(pv) + r" \\", r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def transfer_table(R, caption, label):
    """Cross-domain zero-shot: model trained on the source, evaluated on the target test split."""
    conds = [("tr_gpt2", "GPT-2"), ("tr_learned", "learned"), ("tr_none", "none")]
    pairs = collections.defaultdict(lambda: collections.defaultdict(list))
    for (ds, t, H), runs in R.items():
        if t in dict(conds):
            for r in runs:
                for tgt, m in (r.get("transfer") or {}).items():
                    pairs[(ds, tgt)][t].append(m["mse"])
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l ccc}", r"\toprule", r"Source $\rightarrow$ target & " + " & ".join(n for _, n in conds) + r" \\", r"\midrule"]
    for (src, tgt), d in sorted(pairs.items()):
        vals = [np.mean(d[t]) if d[t] else None for t, _ in conds]
        lines.append(f"{src} $\\rightarrow$ {tgt} & " + " & ".join(fmt_rank(vals, 3)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, nargs="+", help="one or more results roots (merged)")
    p.add_argument("--out", required=True)
    p.add_argument("--bench", default=None, help="JSON written by benchmark_efficiency.py")
    a = p.parse_args()
    R = {}
    for root in sorted(a.root, key=str.lower):  # merge roots in a fixed order; a (dataset, model, horizon, seed) run is counted once
        for k, v in load(root).items():
            seen = {r["seed"] for r in R.get(k, [])}
            R.setdefault(k, []).extend(r for r in v if r["seed"] not in seen)
    os.makedirs(a.out, exist_ok=True)
    w = lambda name, txt: open(os.path.join(a.out, name), "w").write(txt + "\n")  # noqa: E731
    w("main.tex", main_table(R, MAIN_MODELS, "Forecasting results (test MSE/MAE, mean of three seeds; z-scored scale). "
      "All trained models are re-run with the same splits, windows and validation-based model selection; foundation "
      "models are zero-shot with the same context as the look-back of the other models (96 steps; 36 for ILI). $^\\ast$TimesFM-3 forecasts are clipped to the range of their context extended by that range on each side, because without this guard some forecasts explode on near-constant series (Section~\\ref{sec:results}). \\first{Coral}: best; \\runnerup{plum}: second; shaded rows: average over horizons.",
      "tab:main_results"))
    w("main_full.tex", full_table(R, MAIN_MODELS, "Test MSE (mean $\\pm$ standard deviation over three seeds).",
      "tab:detailed"))
    w("llm.tex", paired_table(R, "llm_none", [
        ("llm_gpt2", "GPT-2 (124M, 2019), standardised prompt (PACT)"),
        ("llm_qwen3", "Qwen3-0.6B-Base (2025), same prompt"),
        ("llm_sundial", "Sundial (128M), time-series foundation model"),
        ("llm_window", "learned linear encoding of the window (no LLM)"),
        ("llm_randinit", "randomly initialised GPT-2, same prompt"), ("llm_shuffled", "GPT-2, mismatched window"),
        ("llm_gpt2raw", "GPT-2, raw-value prompt"),
        ("llm_varstatic", "GPT-2, variable description"), ("llm_varnorm", "GPT-2, description + window"),
        ("llm_learned", "learned vectors (same adapter)"), ("llm_random", "fixed random vectors")],
        "What does the anchor contribute, and which source should it come from? Relative change of test MSE with "
        "respect to the same model without anchor (negative: the anchor helps), averaged over horizons and seeds "
        "(high-dimensional datasets: first and third horizon); $p$: Wilcoxon signed-rank test over all cells. "
        "$^\\dagger$: the dataset is very likely part of the pre-training corpus of Sundial "
        "(Section~\\ref{sec:anchor_source}); these cells are excluded from All and $p$.", "tab:llm",
        flagged={"llm_sundial": SUNDIAL_SEEN}))
    abl_caption = ("Component ablation: relative change of test MSE with respect to the full model (positive = the "
                   "component helps); $p$: Wilcoxon signed-rank test over all (dataset, horizon, seed) cells.")
    w("ablation.tex", paired_table(R, "abl_full", [
        ("abl_no_decomp", "w/o decomposition"), ("abl_decomp_ma", "moving-average decomposition"),
        ("abl_no_pdpl", "w/o PDPL (anchor only)"), ("abl_xattn", "BDCA $\\rightarrow$ one-way cross-attention"),
        ("abl_no_bdca", "w/o BDCA (prompt unused)"), ("abl_no_all", "w/o decomposition, PDPL and anchor")],
        abl_caption + " The remaining variants are in Table~S12 of the supplementary material.", "tab:ablation"))
    w("ablation_full.tex", paired_table(R, "abl_full", [
        ("abl_no_decomp", "w/o decomposition"), ("abl_decomp_ma", "moving-average decomposition"),
        ("abl_decomp_fixed", "fixed-bandwidth Gaussian"), ("abl_no_pdpl", "w/o PDPL (anchor only)"),
        ("abl_xattn", "BDCA $\\rightarrow$ one-way cross-attention"), ("abl_no_bdca", "w/o BDCA (prompt unused)"),
        ("abl_bdca2", "BDCA with $I=2$"), ("abl_no_div", "w/o $\\mathcal{L}_{div}$"),
        ("abl_no_align", "w/o $\\mathcal{L}_{align}$"), ("abl_no_freq", "w/o $\\mathcal{L}_{freq}$"),
        ("abl_no_all", "w/o decomposition, PDPL and anchor")], abl_caption, "tab:ablation_full"))
    w("decouple.tex", paired_table(R, "abl_full", [
        ("abl_no_bdca", "w/o prompt pathway (fewer parameters)"),
        ("abl_bdca_self", "w/o prompt pathway, same depth (self-attention blocks)"),
        ("abl_const_prompt", "constant prompt shared by all variates, same capacity"),
        ("llm_none", "prototype prompt without anchor"),
        ("abl_xattn", "one-way cross-attention (fewer parameters)"),
        ("abl_bdca_fwd", "one-way cross-attention, same depth"),
        ("abl_inject_add", "prompt added to the features (fewer parameters)"),
        ("abl_add_deep", "prompt added to the features, same depth"),
        ("abl_adapter_self", "anchor adapter without exchange between variates"),
        ("llm_window", "learned linear window encoding instead of GPT-2"),
        ("llm_window_self", "same, adapter without exchange between variates")],
        "Decoupling the prompt pathway on the high-dimensional datasets (first and third horizon, three seeds): "
        "relative change of test MSE with respect to PACT (positive: PACT is better). ``Same depth'' and ``same "
        "capacity'' variants have the number of parameters of PACT within 5\\%; $p$: Wilcoxon signed-rank test.",
        "tab:decouple", datasets=["Solar", "PEMS08", "ECL", "Traffic"]))
    w("fewshot.tex", fewshot_table(R, "Few-shot forecasting: test MSE averaged over datasets, horizons and seeds when only "
      "the first $p\\%$ of the training windows are used.", "tab:fewshot"))
    w("transfer.tex", transfer_table(R, "Cross-domain zero-shot forecasting (test MSE on the target, averaged over horizons "
      "and seeds): the model is trained on the source dataset only.", "tab:transfer"))
    long_models = [("L512_pact", "PACT"), ("L512_patchtst", "PatchTST"), ("L512_rlinear", "RLinear"),
                   ("zs_timesfm-3.0-pytorch_ctx512", "TimesFM-3"), ("zs_Toto-2.0-313m_ctx512", "Toto-2")]
    # Time-MoE is not run with a 512-step context (cost, see the efficiency figure)
    w("long.tex", main_table(R, long_models, "Long look-back: trained models with $L=512$ and foundation models with a "
      "512-step context, on the same test targets as the main results (Time-MoE is not run with this context because of its "
      "cost).", "tab:long",
      datasets=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]))
    w("prompt_content.tex", paired_table(R, "llm_gpt2", [
        ("llm_gpt2raw", "raw values, integers (TimeCMA template)"), ("llm_raw1f", "raw values, one decimal"),
        ("llm_gz", "training z-score (level kept, units removed)"),
        ("llm_ctr", "window-centred (level removed, units kept)")],
        "What in a raw-value prompt misleads the anchor? Relative change of test MSE with respect to the "
        "window-standardised prompt of PACT (level and units removed), averaged over horizons and seeds; "
        "$p$: Wilcoxon signed-rank test over all cells.", "tab:prompt_content"))
    if a.bench:
        w("efficiency.tex", bench_table(a.bench, "Cost of every model measured on the same idle GPU (ETTm1, $H=96$, "
          "batch 32). PACT additionally needs a one-off offline pass of GPT-2 over the prompts.", "tab:efficiency"))
    else:
      w("efficiency.tex", efficiency_table(R, MAIN_MODELS, "ETTm1", 96,
      "Efficiency on ETTm1 ($H=96$, measured in the same environment). PACT additionally needs a one-off offline "
      "pass of GPT-2 over the prompts (reported in the text).", "tab:efficiency"))
    w("versus.tex", versus_table(R, "pact", [(t, n) for t, n in MAIN_MODELS[1:] if not t.startswith("zs_")],
      "PACT against each baseline: relative difference of test MSE (negative: PACT is better), paired over horizons "
      "and seeds; Wilcoxon signed-rank $p$ in parentheses.", "tab:versus"))
    w("valtest.tex", val_test_table(a.root, "Rank correlation between validation and test MSE over all trained runs "
      "(every model, configuration and seed) of each dataset and horizon. A low or negative $\\rho$ means that "
      "validation-based selection, the only selection used in this paper, cannot be expected to pick the "
      "configuration that is best on the test period.", "tab:valtest"))
    print(f"tables written to {a.out}; {len(R)} (dataset, model, horizon) cells")


if __name__ == "__main__":
    main()
