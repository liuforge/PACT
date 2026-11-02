"""Emit one train.py command per line for an experiment group; feed the file to tools/queue.py.

  python tools/make_jobs.py --group pilot > jobs/pilot.txt
  python tools/make_jobs.py --group main --seeds 2024,2025,2026 > jobs/main.txt

Common flags (--root_path, --emb_root, --out_root, ...) are appended via --extra.
"""
import argparse
import itertools

# look-back, horizons and per-dataset settings shared by all groups
DATASETS = {
    "ETTh1": dict(L=96, H=[96, 192, 336, 720], bs=32),
    "ETTh2": dict(L=96, H=[96, 192, 336, 720], bs=32),
    "ETTm1": dict(L=96, H=[96, 192, 336, 720], bs=32),
    "ETTm2": dict(L=96, H=[96, 192, 336, 720], bs=32),
    "Weather": dict(L=96, H=[96, 192, 336, 720], bs=32),
    "ECL": dict(L=96, H=[96, 192, 336, 720], bs=16),
    "ILI": dict(L=36, H=[24, 36, 48, 60], bs=16, extra="--patch_len 6 --stride 3 --kernel_size 7"),
}

# official TimeCMA scripts (github ChenxiLiu-HNU/TimeCMA, scripts/*.sh): H -> (lr, channel, e_layer, d_layer, dropout, batch)
TIMECMA_OFFICIAL = {
    "ETTh1": {96: (1e-4, 64, 1, 2, 0.7, 16), 192: (1e-4, 64, 1, 2, 0.7, 16), 336: (1e-4, 64, 1, 2, 0.7, 16), 720: (1e-4, 32, 2, 2, 0.8, 16)},
    "ETTh2": {h: (1e-4, 64, 2, 2, 0.3, 16) for h in (96, 192, 336, 720)},
    "ETTm1": {96: (1e-4, 64, 2, 2, 0.5, 16), 192: (1e-4, 64, 2, 2, 0.5, 16), 336: (1e-4, 64, 2, 2, 0.5, 16), 720: (1e-4, 64, 2, 2, 0.7, 16)},
    "ETTm2": {h: (1e-4, 64, 2, 2, 0.3, 16) for h in (96, 192, 336, 720)},
    "Weather": {96: (1e-3, 64, 6, 2, 0.1, 32), 192: (1e-4, 32, 1, 2, 0.1, 32), 336: (1e-4, 32, 1, 2, 0.1, 32), 720: (1e-4, 32, 1, 1, 0.1, 32)},
    "ECL": {96: (1e-3, 128, 3, 6, 0.3, 8), 192: (1e-3, 128, 3, 6, 0.3, 8), 336: (1e-3, 128, 3, 6, 0.1, 8), 720: (1e-3, 128, 3, 6, 0.1, 8)},
    "ILI": {24: (5e-5, 16, 1, 1, 0.1, 16), 36: (1e-4, 16, 1, 1, 0.1, 16), 48: (2.5e-3, 16, 1, 1, 0.3, 16), 60: (1e-4, 32, 1, 1, 0.1, 16)},
}

# ablation variants (relative to the chosen full configuration)
ABLATIONS = {
    "full": "",
    "no_decomp": "--decomp none",
    "decomp_ma": "--decomp moving_avg",
    "decomp_fixed": "--decomp gaussian_fixed",
    "no_pdpl": "--pdpl 0",
    "xattn": "--bdca_iters 0",          # BDCA -> single uni-directional cross-attention
    "no_bdca": "--bdca_iters -1",
    "bdca2": "--bdca_iters 2",
    "no_div": "--lambda_div 0",
    "no_align": "--lambda_align 0",
    "no_freq": "--lambda_freq 0",
    "no_all": "--decomp none --pdpl 0 --llm none",
}
# controls for "does the gain come from the pre-trained LLM?" (R1-6, R3, AE)
LLM_CONTROLS = {
    "llm_gpt2": "--llm stored --emb_tag gpt2",
    "llm_randinit": "--llm stored --emb_tag gpt2-randinit",
    "llm_shuffled": "--llm shuffled --emb_tag gpt2",
    "llm_learned": "--llm learned",
    "llm_random": "--llm random",
    "llm_none": "--llm none",
}


def cmd(ds, H, seed, tag, flags, extra):
    d = DATASETS[ds]
    parts = ["python train.py", f"--dataset {ds} --seq_len {d['L']} --pred_len {H} --seed {seed}",
             f"--batch_size {d['bs']} --tag {tag}", d.get("extra", ""), flags, extra]
    return " ".join(p for p in parts if p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True, choices=["pilot", "main", "timecma", "ablation", "llm", "kproto", "kernel"])
    p.add_argument("--datasets", default=None, help="comma list (default depends on the group)")
    p.add_argument("--seeds", default="2024")
    p.add_argument("--full", default="", help="flags of the chosen full configuration (after the pilot)")
    p.add_argument("--extra", default="")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    pick = lambda default: a.datasets.split(",") if a.datasets else default  # noqa: E731
    jobs = []

    if a.group == "pilot":  # choose token layout and learning rate on a subset, one seed
        for ds, mode, lr in itertools.product(pick(["ETTh1", "ETTm2", "Weather", "ILI"]), ["patch", "variate"], ["1e-4", "5e-4"]):
            for H in (DATASETS[ds]["H"][0], DATASETS[ds]["H"][2]):
                jobs.append(cmd(ds, H, seeds[0], f"pilot_{mode}_lr{lr}", f"--token_mode {mode} --lr {lr}", a.extra))
    elif a.group == "main":
        for ds, s in itertools.product(pick(list(DATASETS)), seeds):
            for H in DATASETS[ds]["H"]:
                jobs.append(cmd(ds, H, s, "pact", a.full, a.extra))
    elif a.group == "timecma":  # official TimeCMA hyper-parameters, re-run under our protocol
        for ds, s in itertools.product(pick(list(DATASETS)), seeds):
            for H in DATASETS[ds]["H"]:
                lr, ch, el, dl, do, bs = TIMECMA_OFFICIAL[ds][H]
                fl = f"--model timecma --d_model {ch} --e_layers {el} --d_layers {dl} --dropout {do} --lr {lr} --warmup 0"
                jobs.append(cmd(ds, H, s, "timecma", fl, a.extra).replace(f"--batch_size {DATASETS[ds]['bs']}", f"--batch_size {bs}"))
    elif a.group == "ablation":
        for ds, s, (tag, fl) in itertools.product(pick(["ETTh2", "ETTm2", "Weather", "ECL", "ILI"]), seeds, ABLATIONS.items()):
            for H in DATASETS[ds]["H"]:
                jobs.append(cmd(ds, H, s, f"abl_{tag}", f"{a.full} {fl}".strip(), a.extra))
    elif a.group == "llm":
        for ds, s, (tag, fl) in itertools.product(pick(["ETTh1", "ETTm2", "Weather", "ECL", "ILI"]), seeds, LLM_CONTROLS.items()):
            for H in DATASETS[ds]["H"]:
                jobs.append(cmd(ds, H, s, tag, f"{a.full} {fl}".strip(), a.extra))
    elif a.group == "kproto":
        for ds, s, k in itertools.product(pick(["Weather", "ETTh2"]), seeds, [4, 8, 16, 32, 64]):
            for H in (DATASETS[ds]["H"][0], DATASETS[ds]["H"][2]):
                jobs.append(cmd(ds, H, s, f"K{k}", f"{a.full} --n_prototypes {k}", a.extra))
    elif a.group == "kernel":
        for ds, s, k in itertools.product(pick(["Weather", "ETTh2"]), seeds, [7, 11, 25, 49]):
            for H in (DATASETS[ds]["H"][0], DATASETS[ds]["H"][2]):
                jobs.append(cmd(ds, H, s, f"ks{k}", f"{a.full} --kernel_size {k}", a.extra))
    print("\n".join(jobs))


if __name__ == "__main__":
    main()
