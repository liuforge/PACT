"""Commands that re-train the main-table configuration of a model with --save_preds 1 (for forecast plots).

The configuration is read from the results.json of the main-table run (for '<model>__<hp>' variants the one with the
lowest seed-2024 validation MSE, as in make_tables.load); the tag becomes 'viz_<model>' so the main runs are untouched.
  python tools/viz_jobs.py --root <results>... --datasets ETTm2,Weather,ECL --models pact,timecma --H 96 > jobs/viz.txt
"""
import argparse
import glob
import json
import os

DERIVED = {"num_vars", "d_llm", "device_str", "transfer", "eval_datasets"}


def pick(roots, ds, model, H, seed):
    best = None
    for root in roots:
        for f in glob.glob(os.path.join(root, ds, f"{model}*", f"L*_H{H}_s{seed}", "results.json")):
            tag = f.split(os.sep)[-3]
            if tag != model and not tag.startswith(model + "__"):
                continue
            r = json.load(open(f))
            if r["config"].get("train_percent", 100) not in (100, None):
                continue
            if best is None or r["val"]["mse"] < best["val"]["mse"]:
                best = r
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", nargs="+", required=True)
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--seed", type=int, default=2024)
    a = p.parse_args()
    for ds in a.datasets.split(","):
        for m in a.models.split(","):
            r = pick(a.root, ds, m, a.H, a.seed)
            if r is None:
                print(f"# missing {ds} {m}")
                continue
            c = dict(r["config"], tag=f"viz_{m}", save_preds=1, overwrite=0)
            args = " ".join(f"--{k} {v}" for k, v in c.items() if k not in DERIVED and v is not None)
            print(f"python train.py {args}")


if __name__ == "__main__":
    main()
