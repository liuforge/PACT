"""Linear probes on stored prompt embeddings: what does each prompt variant tell the forecaster?

For each prompt store (raw TimeCMA prompt, its factorial variants and the standardised prompt) and each dataset:
  level R^2   - ridge regression from the embedding to the window level (mean of the training-z-scored look-back
                window), fitted on training windows and scored on test windows; pooled over variates and per
                variate (one probe per variate, R^2 averaged);
  period acc. - ridge classifier separating validation from test windows (balanced; fitted on the first half
                of each split, scored on the second half), i.e. how well the embedding identifies the period.
No model is trained; the probes only read the frozen GPT-2 embeddings that PACT receives.

  python probe_prompts.py --datasets ETTh1,ETTh2,ETTm2,Weather,ILI --out probe_prompts.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pact.data import load_series  # noqa: E402

TAGS = [("gpt2", "raw (TimeCMA)"), ("gpt2-raw1f", "raw, 1 decimal"), ("gpt2-gz", "train z-score (level kept)"),
        ("gpt2-ctr", "window-centred (units kept)"), ("gpt2-norm", "window-standardised (PACT)")]


def ridge_fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda A: ((A - mu) / sd) @ w + y.mean()


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="ETTh1,ETTh2,ETTm2,Weather,ILI")
    p.add_argument("--root_path", default="data")
    p.add_argument("--emb_root", default="embeddings")
    p.add_argument("--per_split", type=int, default=3000, help="samples per split and variate")
    p.add_argument("--lam", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    out = []
    for ds in a.datasets.split(","):
        L = 36 if ds == "ILI" else 96
        s = load_series(ds, a.root_path, L)
        x = s["scaled"]
        N = x.shape[1]
        b1, b2 = s["border1s"], s["border2s"]
        samples = {}
        for k, name in enumerate(("train", "val", "test")):
            starts = np.arange(b1[k], b2[k] - L + 1)
            n = min(a.per_split * N, len(starts) * N)
            idx = rng.choice(len(starts) * N, size=n, replace=False)
            samples[name] = (starts[idx // N], idx % N)
        level = {k: np.array([x[w:w + L, j].mean() for w, j in zip(*v)]) for k, v in samples.items()}
        for tag, desc in TAGS:
            f = os.path.join(a.emb_root, ds, f"{tag}_L{L}.npy")
            if not os.path.exists(f):
                print(f"skip {ds} {tag}: no store")
                continue
            E = np.load(f, mmap_mode="r")
            feats = {k: np.asarray(E[w, j], dtype=np.float64) for k, (w, j) in samples.items()}
            pred = ridge_fit(feats["train"], level["train"], a.lam)
            lvl = r2(level["test"], pred(feats["test"]))
            per = []
            for j in range(N):
                mtr, mte = samples["train"][1] == j, samples["test"][1] == j
                if mtr.sum() > 50 and mte.sum() > 20:
                    pj = ridge_fit(feats["train"][mtr], level["train"][mtr], a.lam)
                    per.append(r2(level["test"][mte], pj(feats["test"][mte])))
            lvl_var = float(np.mean(per))
            # period: validation vs. test windows, fit on the first half of each split and score on the second
            h = {k: len(feats[k]) // 2 for k in ("val", "test")}
            Xtr = np.concatenate([feats["val"][:h["val"]], feats["test"][:h["test"]]])
            ytr = np.r_[np.zeros(h["val"]), np.ones(h["test"])]
            Xte = np.concatenate([feats["val"][h["val"]:], feats["test"][h["test"]:]])
            yte = np.r_[np.zeros(len(feats["val"]) - h["val"]), np.ones(len(feats["test"]) - h["test"])]
            clf = ridge_fit(Xtr, ytr, a.lam)
            acc = float(((clf(Xte) > 0.5) == yte).mean())
            out.append(dict(dataset=ds, tag=tag, prompt=desc, level_r2_test=lvl, level_r2_per_variate=lvl_var,
                            period_acc=acc))
            print(f"{ds:8s} {desc:30s} level R2 (test) pooled {lvl:+.3f} per-variate {lvl_var:+.3f} | "
                  f"val-vs-test accuracy {acc:.3f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
