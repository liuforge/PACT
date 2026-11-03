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

DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL", "ILI"]
HORIZONS = {"ILI": [24, 36, 48, 60]}
MAIN_MODELS = [  # (tag after selection, column title)
    ("pact", "PACT"), ("timecma", "TimeCMA"), ("patchtst", "PatchTST"), ("itransformer", "iTrans."),
    ("timesnet", "TimesNet"), ("dlinear", "DLinear"), ("rlinear", "RLinear"),
    ("zs_timesfm-3.0-pytorch_ctxL", "TimesFM-3"), ("zs_Toto-2.0-313m_ctxL", "Toto-2"),
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


def fmt_rank(vals, digits=3):
    """Bold best, underline second (ties share the rank)."""
    finite = sorted({round(v, digits) for v in vals if v is not None})
    out = []
    for v in vals:
        if v is None:
            out.append("--")
            continue
        s = f"{v:.{digits}f}"
        r = round(v, digits)
        out.append(r"\textbf{" + s + "}" if r == finite[0] else (r"\underline{" + s + "}" if len(finite) > 1 and r == finite[1] else s))
    return out


def main_table(R, models, caption, label):
    cols = "l l " + " ".join(["cc"] * len(models))
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{" + cols + "}", r"\toprule",
             "& & " + " & ".join(r"\multicolumn{2}{c}{" + t + "}" for _, t in models) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(models))),
             "Dataset & $H$ & " + " & ".join(["MSE & MAE"] * len(models)) + r" \\", r"\midrule"]
    wins = collections.Counter()
    for ds in DATASETS:
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
            if H != "Avg":
                best = min(v for v in mse if v is not None)
                for (tag, _), v in zip(models, mse):
                    if v is not None and round(v, 3) == round(best, 3):
                        wins[tag] += 1
            cells = [f"{a} & {b}" for a, b in zip(fmt_rank(mse), fmt_rank(mae))]
            rows.append(("" if rows else ds) + f" & {H} & " + " & ".join(cells) + r" \\")
        if rows:
            lines += rows + [r"\midrule"]
    lines[-1] = r"\midrule"
    lines.append(r"\multicolumn{2}{l}{1st count (MSE)} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{wins[t]}}}" for t, _ in models) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines)


def full_table(R, models, caption, label):
    """Appendix: mean +- std of MSE for every cell."""
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
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


def paired_table(R, ref, variants, caption, label, datasets=None):
    """Relative test-MSE change of each variant vs. ref over (dataset, H, seed) cells, with Wilcoxon p."""
    cells = collections.defaultdict(dict)
    for (ds, tag, H), runs in R.items():
        if datasets and ds not in datasets:
            continue
        for r in runs:
            cells[(ds, H, r["seed"])][tag] = r["mse"]
    dss = sorted({k[0] for k in cells if ref in cells[k]}, key=DATASETS.index)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l " + "c " * (len(dss) + 2) + "}", r"\toprule",
             "Variant & " + " & ".join(dss) + r" & All & $p$ \\", r"\midrule"]
    for tag, name in variants:
        row = []
        allx, ally = [], []
        for ds in dss:
            ks = [k for k in cells if k[0] == ds and ref in cells[k] and tag in cells[k]]
            if not ks:
                row.append("--")
                continue
            x = np.array([cells[k][ref] for k in ks])
            y = np.array([cells[k][tag] for k in ks])
            allx += list(x)
            ally += list(y)
            row.append(f"{np.mean((y - x) / x) * 100:+.1f}\\%")
        if len(allx) >= 6:
            x, y = np.array(allx), np.array(ally)
            p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            row += [f"{np.mean((y - x) / x) * 100:+.1f}\\%", f"{p:.3f}"]
        else:
            row += ["--", "--"]
        lines.append(f"{name} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def efficiency_table(R, models, ds, H, caption, label):
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
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


def fewshot_table(R, caption, label):
    """Test MSE averaged over datasets, horizons and seeds for each training fraction and anchor condition."""
    conds = [("gpt2", "GPT-2 anchor"), ("learned", "learned vectors"), ("none", "no anchor")]
    pcts = sorted({int(t[2:].split("_")[0]) for (_, t, _) in R if t.startswith("fs") and t[2:].split("_")[0].isdigit()})
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
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
        pv.append(f"{wilcoxon(x, y).pvalue:.2f}" if len(ks) >= 6 else "--")
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
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
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
    a = p.parse_args()
    R = {}
    for root in a.root:  # merge roots; a (dataset, model, horizon, seed) run is counted once
        for k, v in load(root).items():
            seen = {r["seed"] for r in R.get(k, [])}
            R.setdefault(k, []).extend(r for r in v if r["seed"] not in seen)
    os.makedirs(a.out, exist_ok=True)
    w = lambda name, txt: open(os.path.join(a.out, name), "w").write(txt + "\n")  # noqa: E731
    w("main.tex", main_table(R, MAIN_MODELS, "Forecasting results (test MSE/MAE, mean of three seeds; z-scored scale). "
      "All trained models are re-run with the same splits, windows and validation-based model selection; foundation "
      "models are zero-shot with the same context as the look-back of the other models (96 steps; 36 for ILI). \\textbf{Bold}: best; \\underline{underline}: second.",
      "tab:main_results"))
    w("main_full.tex", full_table(R, MAIN_MODELS, "Test MSE (mean $\\pm$ standard deviation over three seeds).",
      "tab:detailed"))
    w("llm.tex", paired_table(R, "llm_none", [
        ("llm_gpt2", "GPT-2, standardised prompt (PACT)"), ("llm_gpt2raw", "GPT-2, raw-value prompt"),
        ("llm_randinit", "random-init GPT-2"), ("llm_shuffled", "GPT-2, mismatched window"),
        ("llm_varstatic", "GPT-2, variable description"), ("llm_varnorm", "GPT-2, description + window"),
        ("llm_learned", "learned vectors (same adapter)"), ("llm_random", "fixed random vectors")],
        "Where does the anchor's contribution come from? Relative change of test MSE with respect to the same model "
        "without any anchor, averaged over horizons and seeds; $p$: Wilcoxon signed-rank test over all cells.",
        "tab:llm"))
    w("ablation.tex", paired_table(R, "abl_full", [
        ("abl_no_decomp", "w/o decomposition"), ("abl_decomp_ma", "moving-average decomposition"),
        ("abl_decomp_fixed", "fixed-bandwidth Gaussian"), ("abl_no_pdpl", "w/o PDPL (anchor only)"),
        ("abl_xattn", "BDCA $\\rightarrow$ one-way cross-attention"), ("abl_no_bdca", "w/o BDCA"),
        ("abl_bdca2", "BDCA with $I=2$"), ("abl_no_div", "w/o $\\mathcal{L}_{div}$"),
        ("abl_no_align", "w/o $\\mathcal{L}_{align}$"), ("abl_no_freq", "w/o $\\mathcal{L}_{freq}$"),
        ("abl_no_all", "w/o decomposition, PDPL and anchor")],
        "Component ablation: relative change of test MSE with respect to the full model (positive = the component "
        "helps); $p$: Wilcoxon signed-rank test over all (dataset, horizon, seed) cells.", "tab:ablation"))
    w("fewshot.tex", fewshot_table(R, "Few-shot forecasting: test MSE averaged over datasets, horizons and seeds when only "
      "the first $p\\%$ of the training windows are used.", "tab:fewshot"))
    w("transfer.tex", transfer_table(R, "Cross-domain zero-shot forecasting (test MSE on the target, averaged over horizons "
      "and seeds): the model is trained on the source dataset only.", "tab:transfer"))
    long_models = [("L512_pact", "PACT"), ("L512_patchtst", "PatchTST"), ("L512_rlinear", "RLinear"),
                   ("zs_timesfm-3.0-pytorch_ctx512", "TimesFM-3"), ("zs_Toto-2.0-313m_ctx512", "Toto-2"),
                   ("zs_TimeMoE-200M_ctx512", "Time-MoE")]
    w("long.tex", main_table(R, long_models, "Long look-back: trained models with $L=512$ and foundation models with a "
      "512-step context, on the same test targets as Table~\\ref{tab:main_results}.", "tab:long"))
    w("efficiency.tex", efficiency_table(R, MAIN_MODELS, "ETTm1", 96,
      "Efficiency on ETTm1 ($H=96$, measured in the same environment). PACT additionally needs a one-off offline "
      "pass of GPT-2 over the prompts (reported in the text).", "tab:efficiency"))
    print(f"tables written to {a.out}; {len(R)} (dataset, model, horizon) cells")


if __name__ == "__main__":
    main()
