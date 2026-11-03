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

# Time-Series-Library baselines: architecture settings from TSLib scripts (ETT) or the models' own
# repositories; training follows TSLib (10 epochs, patience 3, lr halved every epoch).
TSLIB_TRAIN = "--sched type1 --epochs 10 --patience 3 --warmup 0"
_PT_ETT = {  # PatchTST, TSLib ETT scripts: H -> (e_layers, n_heads)
    "ETTh1": {96: (1, 2), 192: (1, 8), 336: (1, 8), 720: (1, 16)},
    "ETTh2": {h: (3, 4) for h in (96, 192, 336, 720)},
    "ETTm1": {96: (1, 2), 192: (3, 2), 336: (1, 4), 720: (3, 4)},
    "ETTm2": {96: (3, 16), 192: (3, 2), 336: (1, 4), 720: (3, 4)},
}
_TN = {"ETTh1": (16, 32), "ETTh2": (32, 32), "ETTm1": (64, 64), "ETTm2": (32, 32),  # TimesNet (d_model, d_ff)
       "ECL": (256, 512), "Weather": (32, 32), "ILI": (768, 768)}
_IT = {"ETTh1": (256, 256, 2), "ETTh2": (128, 128, 2), "ETTm1": (128, 128, 2), "ETTm2": (128, 128, 2),  # iTransformer
       "ECL": (512, 512, 3), "Weather": (512, 512, 3), "ILI": (128, 128, 2)}


def tslib_flags(model, ds, H):
    lr, extra = "1e-4", ""
    if model == "patchtst":
        if ds in _PT_ETT:
            e, nh = _PT_ETT[ds][H]
            arch = f"--e_layers {e} --n_heads {nh} --d_model 512 --d_ff 2048"
        elif ds == "ILI":
            arch = "--e_layers 3 --n_heads 4 --d_model 16 --d_ff 128 --dropout 0.3"
        else:
            arch = "--e_layers 3 --n_heads 16 --d_model 128 --d_ff 256 --dropout 0.2"
    elif model == "itransformer":
        dm, dff, e = _IT[ds]
        arch = f"--e_layers {e} --n_heads 8 --d_model {dm} --d_ff {dff}"
        lr = "5e-4" if ds == "ECL" else lr
    elif model == "timesnet":
        dm, dff = _TN[ds]
        arch = f"--e_layers 2 --d_model {dm} --d_ff {dff} --top_k 5 --num_kernels 6"
    else:  # dlinear / rlinear: no architecture settings
        arch = ""
    label = "--label_len 18" if ds == "ILI" else "--label_len 48"
    return f"--model {model} {arch} {label} --lr {lr} {TSLIB_TRAIN} {extra}".replace("  ", " ").strip()


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
    p.add_argument("--group", required=True, choices=["pilot", "pilot2", "long", "main", "timecma", "v5", "tslib", "ablation", "llm", "kproto", "kernel"])
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
    elif a.group == "pilot2":  # capacity / regularisation on the small datasets (patch layout), one seed
        for ds in pick(["ETTh1", "ETTh2", "ETTm1", "ILI"]):
            lr = "5e-4" if ds == "ILI" else "1e-4"
            for dm, do, ffn in itertools.product([64, 128], [0.1, 0.3], [0, 1]):
                for H in (DATASETS[ds]["H"][0], DATASETS[ds]["H"][2]):
                    fl = f"--token_mode patch --lr {lr} --d_model {dm} --d_ff {2 * dm} --dropout {do} --bdca_ffn {ffn}"
                    jobs.append(cmd(ds, H, seeds[0], f"pilot2_d{dm}_do{do}_ffn{ffn}", fl, a.extra))
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
    elif a.group == "v5":  # first-submission model; settings of the original run scripts (run_ablation_*.sh)
        fl = ("--model v5 --d_model 64 --e_layers 2 --d_layers 2 --kernel_size 25 --n_prototypes 16 "
              "--v5_encoder conv_nomask_hybrid --v5_fusion weighted --lr 5e-4 --dropout 0.1 --lambda_freq 0")
        for ds, s in itertools.product(pick(list(DATASETS)), seeds):
            for H in DATASETS[ds]["H"]:
                jobs.append(cmd(ds, H, s, "v5", fl, a.extra).replace("--kernel_size 7 ", ""))
    elif a.group == "tslib":
        for ds, s in itertools.product(pick(list(DATASETS)), seeds):
            for H in DATASETS[ds]["H"]:
                for m in ("patchtst", "itransformer", "timesnet"):
                    jobs.append(cmd(ds, H, s, m, tslib_flags(m, ds, H), a.extra))
                for m, lr in itertools.product(("dlinear", "rlinear"), ("1e-4", "1e-3", "5e-3")):
                    fl = tslib_flags(m, ds, H).replace("--lr 1e-4", f"--lr {lr}")
                    jobs.append(cmd(ds, H, s, f"{m}__lr{lr}", fl, a.extra))
    elif a.group == "long":  # L=512 (vs foundation models with a 512 context); the prompt covers the last 96 steps
        for ds, s in itertools.product(pick(["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "ECL"]), seeds):
            for H in DATASETS[ds]["H"]:
                c = lambda tag, fl: cmd(ds, H, s, tag, fl, a.extra).replace(f"--seq_len {DATASETS[ds]['L']}", "--seq_len 512")  # noqa: E731
                for lr in ("1e-4", "5e-4"):
                    jobs.append(c(f"L512_pact__lr{lr}", f"{a.full} --prompt_len 96 --lr {lr}".strip()))
                jobs.append(c("L512_patchtst", tslib_flags("patchtst", ds, H)))
                for lr in ("1e-4", "1e-3", "5e-3"):
                    jobs.append(c(f"L512_rlinear__lr{lr}", tslib_flags("rlinear", ds, H).replace("--lr 1e-4", f"--lr {lr}")))
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
