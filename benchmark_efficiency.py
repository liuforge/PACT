"""Measure the cost of every model on the same idle GPU (ETTm1, H=96, batch 32).

Trained models: trainable parameters, time per training step (after warm-up) and the implied time per epoch,
inference latency per batch and peak GPU memory during training. Foundation models: parameters, inference
latency and peak memory (zero-shot, contexts 96 and 512). Writes a JSON consumed by the tables and figures.

  python benchmark_efficiency.py --root_path data --out efficiency_ETTm1.json [--fm 1]
(timing only: TimeCMA is fed the same-shaped standardised-prompt embeddings, which does not change its cost)
"""
import argparse
import json
import os
import shlex
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402
from pact.data import EmbeddingStore, build_datasets, load_series  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
from make_jobs import TIMECMA_OFFICIAL, tslib_flags  # noqa: E402


def configs(pact_hp):
    lr, d, ffn = pact_hp
    lr_, ch, el, dl, do, bs = TIMECMA_OFFICIAL["ETTm1"][96]
    return {
        "PACT": f"--model pact --token_mode patch --dropout 0.1 --llm stored --emb_tag gpt2-norm --lr {lr} --d_model {d} --d_ff {2 * d} --bdca_ffn {ffn}",
        "PACT (no anchor)": f"--model pact --token_mode patch --dropout 0.1 --llm none --lr {lr} --d_model {d} --d_ff {2 * d} --bdca_ffn {ffn}",
        "TimeCMA": f"--model timecma --emb_tag gpt2-norm --d_model {ch} --e_layers {el} --d_layers {dl} --dropout {do}",
        "PatchTST": tslib_flags("patchtst", "ETTm1", 96),
        "iTransformer": tslib_flags("itransformer", "ETTm1", 96),
        "TimesNet": tslib_flags("timesnet", "ETTm1", 96),
        "DLinear": tslib_flags("dlinear", "ETTm1", 96),
        "RLinear": tslib_flags("rlinear", "ETTm1", 96),
    }


def bench_trained(name, flags, a, device):
    args = train.get_args(shlex.split(f"--dataset ETTm1 --root_path {a.root_path} --seq_len 96 --pred_len 96 "
                                      f"--batch_size 32 --emb_root {a.emb_root} {flags}"))
    series = load_series("ETTm1", args.root_path, 96)
    args.num_vars = series["raw"].shape[1]
    store = None
    if args.model in train.NEEDS_EMB and args.llm == "stored":
        store = EmbeddingStore(os.path.join(args.emb_root, "ETTm1", f"{args.emb_tag}_L96.npy"), in_memory=True)
        args.d_llm = store.d_llm
    else:
        args.d_llm = 768
        if args.model not in train.NEEDS_EMB:
            args.llm = "none"
    args.prompt_len = 96
    args.device_str = str(device)
    tr, _, te = build_datasets(series, 96, 96, store)
    model = train.MODELS[args.model](args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loader = torch.utils.data.DataLoader(tr, batch_size=32, shuffle=True, drop_last=True)
    it = iter(loader)
    torch.cuda.reset_peak_memory_stats(device)

    def step():
        nonlocal it
        try:
            x, y, _, emb, xm = next(it)
        except StopIteration:
            it = iter(loader)
            x, y, _, emb, xm = next(it)
        x, y, xm = x.to(device), y.to(device), xm.to(device)
        emb = emb.to(device) if emb.numel() else None
        model.train()
        pred, aux = model(x, emb, xm)
        loss = F.mse_loss(pred, y) + sum(v for v in aux.values()) * 0.01
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(a.steps):
        step()
    torch.cuda.synchronize()
    per_step = (time.perf_counter() - t0) / a.steps
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20
    model.eval()
    tl = torch.utils.data.DataLoader(te, batch_size=32, shuffle=False, drop_last=True)
    lat = []
    with torch.no_grad():
        for i, (x, _, _, emb, xm) in enumerate(tl):
            if i >= a.infer_batches + 5:
                break
            x, xm = x.to(device), xm.to(device)
            emb = emb.to(device) if emb.numel() else None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x, emb, xm)
            torch.cuda.synchronize()
            if i >= 5:
                lat.append(time.perf_counter() - t0)
    return dict(model=name, params=n_params, s_per_step=per_step, s_per_epoch=per_step * len(tr) / 32,
                infer_ms_per_batch=float(np.mean(lat) * 1000), peak_mem_mib=peak)


def bench_fm(name, cls, ckpt, ctx, a, device):
    import eval_foundation as ef
    torch.cuda.reset_peak_memory_stats(device)
    m = getattr(ef, cls)(ckpt, device)
    net = getattr(m.model, "model", m.model)
    n_params = sum(p.numel() for p in net.parameters()) if isinstance(net, torch.nn.Module) else None
    series = load_series("ETTm1", a.root_path, 96)
    data = series["scaled"]
    starts = np.arange(series["border1s"][2] + 96, series["border1s"][2] + 96 + 32 * (a.infer_batches + 5), 1)
    lat = []
    for i in range(a.infer_batches + 5):
        t = starts[i * 32:(i + 1) * 32]
        ctxa = np.stack([data[s - ctx:s] for s in t]).astype(np.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m(ctxa, 96)
        torch.cuda.synchronize()
        if i >= 5:
            lat.append(time.perf_counter() - t0)
    return dict(model=f"{name} (ctx {ctx})", params=n_params, s_per_step=None, s_per_epoch=None,
                infer_ms_per_batch=float(np.mean(lat) * 1000), peak_mem_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", default="data")
    p.add_argument("--emb_root", default="embeddings")
    p.add_argument("--pact_hp", default="1e-4,128,0", help="lr,d_model,ffn chosen for ETTm1-96")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--infer_batches", type=int, default=50)
    p.add_argument("--fm", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device("cuda")
    torch.manual_seed(0)
    lr, d, f = a.pact_hp.split(",")
    out = []
    for name, flags in configs((lr, int(d), int(f))).items():
        out.append(bench_trained(name, flags, a, device))
        print(json.dumps(out[-1]), flush=True)
        torch.cuda.empty_cache()
    if a.fm:
        for name, cls, ckpt in (("TimesFM-3", "TimesFM3", "hf_models/timesfm-3.0-pytorch"),
                                ("Toto-2 22M", "Toto2", "hf_models/Toto-2.0-22m"),
                                ("Toto-2 313M", "Toto2", "hf_models/Toto-2.0-313m"),
                                ("Time-MoE 50M", "TimeMoE", "hf_models/TimeMoE-50M"),
                                ("Time-MoE 200M", "TimeMoE", "hf_models/TimeMoE-200M")):
            for ctx in (96, 512):
                try:
                    out.append(bench_fm(name, cls, ckpt, ctx, a, device))
                    print(json.dumps(out[-1]), flush=True)
                except Exception as e:  # report and continue
                    print(f"{name} ctx {ctx} failed: {e}", flush=True)
                torch.cuda.empty_cache()
    json.dump(dict(device=torch.cuda.get_device_name(0), results=out), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
