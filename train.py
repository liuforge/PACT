"""Train and evaluate one PACT configuration and write a self-contained results.json.

Protocol (fixes relative to the original TimeCMA-derived scripts):
  * the checkpoint is selected on validation MSE only; the test split is evaluated once at the end;
  * the training loader is shuffled; val/test loaders keep every window (drop_last=False);
  * metrics are reported on the z-scored scale (the LTSF convention) and additionally on the raw scale.
"""
import argparse
import json
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pact.data import EmbeddingStore, build_datasets, load_series
from pact.baselines import TSLib, V5, RLinear, TimeCMA
from pact.model import PACT

MODELS = {"pact": PACT, "timecma": TimeCMA, "v5": V5, "rlinear": RLinear,
          "patchtst": TSLib, "itransformer": TSLib, "dlinear": TSLib, "timesnet": TSLib}
NEEDS_EMB = {"pact", "timecma", "v5"}


def get_args(argv=None):
    p = argparse.ArgumentParser()
    a = p.add_argument
    # data
    a("--dataset", required=True)
    a("--root_path", default="./data")
    a("--data_file", default=None, help="csv path relative to root_path (overrides the registry)")
    a("--seq_len", type=int, default=96)
    a("--pred_len", type=int, default=96)
    a("--emb_root", default="./embeddings")
    a("--emb_tag", default="gpt2", help="embedding store name: {emb_root}/{dataset}/{emb_tag}_L{seq_len}.npy")
    # model
    a("--model", default="pact", choices=list(MODELS))
    a("--d_layers", type=int, default=1, help="TimeCMA decoder layers")
    a("--tc_d_ff", type=int, default=32, help="TimeCMA cross-modal FFN width")
    a("--v5_encoder", default="conv_nomask_hybrid", choices=["conv_transformer_unified", "conv_nomask_hybrid", "transformer_nomask"])
    a("--v5_fusion", default="weighted", choices=["simple", "attention", "gating", "weighted"])
    a("--label_len", type=int, default=48, help="TSLib models")
    a("--factor", type=int, default=3)
    a("--top_k", type=int, default=5)
    a("--num_kernels", type=int, default=6)
    a("--moving_avg", type=int, default=25)
    a("--token_mode", default="patch", choices=["patch", "variate"])
    a("--patch_len", type=int, default=16)
    a("--stride", type=int, default=8)
    a("--d_model", type=int, default=128)
    a("--n_heads", type=int, default=8)
    a("--e_layers", type=int, default=1)
    a("--d_ff", type=int, default=256)
    a("--dropout", type=float, default=0.1)
    a("--head_dropout", type=float, default=0.0)
    a("--revin", type=int, default=1)
    a("--revin_affine", type=int, default=0)
    a("--decomp", default="gaussian", choices=["gaussian", "gaussian_fixed", "moving_avg", "none"])
    a("--kernel_size", type=int, default=25)
    a("--init_sigma", type=float, default=None)
    a("--llm", default="stored", choices=list(PACT.LLM_MODES))
    a("--adapter_dim", type=int, default=32)
    a("--adapter_heads", type=int, default=4)
    a("--pdpl", type=int, default=1)
    a("--n_prototypes", type=int, default=16)
    a("--bdca_iters", type=int, default=1, help="-1: no BDCA, 0: single cross-attention, I>=1: I bi-directional rounds")
    a("--bdca_ffn", type=int, default=1)
    a("--fusion", default="scalar", choices=["scalar", "variate", "sum"])
    # loss
    a("--lambda_freq", type=float, default=0.1)
    a("--lambda_align", type=float, default=0.05)
    a("--lambda_div", type=float, default=0.01)
    a("--tau_init", type=float, default=0.1)
    # optimisation
    a("--batch_size", type=int, default=32)
    a("--lr", type=float, default=1e-4)
    a("--weight_decay", type=float, default=1e-3)
    a("--epochs", type=int, default=50)
    a("--patience", type=int, default=10)
    a("--sched", default="cosine", choices=["cosine", "type1"], help="type1: TSLib halving every epoch")
    a("--warmup", type=float, default=0.1, help="fraction of total steps used for linear warm-up (cosine)")
    a("--clip", type=float, default=5.0)
    a("--seed", type=int, default=2024)
    a("--num_workers", type=int, default=4)
    a("--device", default="cuda")
    a("--amp", type=int, default=0, help="bf16 autocast")
    # bookkeeping
    a("--out_root", default="./results")
    a("--tag", default="pact", help="experiment name; results go to out_root/dataset/tag/")
    a("--overwrite", type=int, default=0)
    a("--save_ckpt", type=int, default=0)
    a("--save_preds", type=int, default=0)
    return p.parse_args(argv)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_dir(args):
    return os.path.join(args.out_root, args.dataset, args.tag, f"L{args.seq_len}_H{args.pred_len}_s{args.seed}")


def git_rev():
    try:
        root = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def freq_loss(pred, y):
    return (torch.fft.rfft(pred, dim=1, norm="ortho").abs() - torch.fft.rfft(y, dim=1, norm="ortho").abs()).abs().mean()


def aux_weight(name, args):
    if name == "legacy":
        return 1.0
    return args.lambda_align if name.startswith("align") else args.lambda_div


@torch.no_grad()
def evaluate(model, loader, device, args, series=None, timing=False):
    model.eval()
    preds, trues = [], []
    batch_times = []
    for x, y, _, emb, xm in loader:
        x, y, xm = x.to(device), y.to(device), xm.to(device)
        emb = emb.to(device) if emb.numel() else None
        if timing and device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(args.amp)):
            out, _ = model(x, emb, xm)
        if timing and device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - t0)
        preds.append(out.float().cpu())
        trues.append(y.cpu())
    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()
    res = dict(mse=float(np.mean((pred - true) ** 2)), mae=float(np.mean(np.abs(pred - true))),
               n_windows=int(pred.shape[0]))
    if series is not None:  # raw (original-unit) scale
        pr = pred * series["std"] + series["mean"]
        tr = true * series["std"] + series["mean"]
        res.update(mse_raw=float(np.mean((pr - tr) ** 2)), mae_raw=float(np.mean(np.abs(pr - tr))))
    if timing:
        res["infer_s_per_batch"] = float(np.mean(batch_times[1:] if len(batch_times) > 1 else batch_times))
    return res, pred, true


def main(argv=None):
    args = get_args(argv)
    out_dir = run_dir(args)
    res_path = os.path.join(out_dir, "results.json")
    if os.path.exists(res_path) and not args.overwrite:
        print(f"[skip] {res_path} exists")
        return json.load(open(res_path))
    os.makedirs(out_dir, exist_ok=True)
    if args.model != "pact":  # baselines are trained with their own (MSE-only) objective
        args.lambda_freq = args.lambda_align = args.lambda_div = 0.0
        if args.model in NEEDS_EMB and args.llm not in ("stored", "shuffled"):
            raise ValueError(f"{args.model} needs stored LLM embeddings")
        if args.model not in NEEDS_EMB:
            args.llm = "none"
    seed_all(args.seed)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    series = load_series(args.dataset, args.root_path, args.seq_len, args.data_file)
    args.num_vars = series["raw"].shape[1]

    store = None
    if args.llm in ("stored", "shuffled"):
        path = os.path.join(args.emb_root, args.dataset, f"{args.emb_tag}_L{args.seq_len}.npy")
        if os.path.exists(path + ".progress") and not json.load(open(path + ".progress"))["finished"]:
            raise RuntimeError(f"embedding store {path} is still being built")
        store = EmbeddingStore(path)
        n_expected = series["raw"].shape[0] - args.seq_len + 1
        if len(store) != n_expected or store.arr.shape[1] != args.num_vars:
            raise ValueError(f"embedding store {path} has shape {store.arr.shape}, expected [{n_expected}, {args.num_vars}, d]")
        if args.llm == "shuffled":  # pre-trained embeddings of a *different* window: breaks sample alignment
            store.index_map = np.random.default_rng(args.seed).permutation(len(store))
        args.d_llm = store.d_llm
    else:
        args.d_llm = getattr(args, "d_llm", None) or 768

    train_set, val_set, test_set = build_datasets(series, args.seq_len, args.pred_len, store)
    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=args.num_workers > 0)
    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, generator=g, **kw)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **kw)

    args.device_str = str(device)
    model = MODELS[args.model](args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.dataset} L={args.seq_len} H={args.pred_len} seed={args.seed} tag={args.tag} | "
          f"params={n_params:,} | windows train/val/test={len(train_set)}/{len(val_set)}/{len(test_set)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warm = max(1, int(args.warmup * total_steps))
    steps_per_epoch = len(train_loader)
    if args.sched == "type1":
        lr_lambda = lambda s: 0.5 ** (s // steps_per_epoch)  # noqa: E731
    else:
        lr_lambda = lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm)))  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ckpt_path = os.path.join(out_dir, "best.pt")
    best_state, best_val, best_epoch, bad = None, float("inf"), -1, 0
    history, epoch_times = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        sums = {}
        n_batches = 0
        for x, y, _, emb, xm in train_loader:
            x, y, xm = x.to(device, non_blocking=True), y.to(device, non_blocking=True), xm.to(device, non_blocking=True)
            emb = emb.to(device, non_blocking=True) if emb.numel() else None
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(args.amp)):
                pred, aux = model(x, emb, xm)
            pred = pred.float()
            mse = F.mse_loss(pred, y)
            loss = mse
            terms = {"mse": mse}
            if args.lambda_freq > 0:
                terms["freq"] = freq_loss(pred, y)
                loss = loss + args.lambda_freq * terms["freq"]
            for k, v in aux.items():
                loss = loss + aux_weight(k, args) * v.float()
                terms[k] = v
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            sched.step()
            for k, v in terms.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            n_batches += 1
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - t0)
        val, _, _ = evaluate(model, val_loader, device, args)
        rec = {"epoch": epoch, "time_s": epoch_times[-1], "val_mse": val["mse"], "val_mae": val["mae"],
               **{f"train_{k}": v / n_batches for k, v in sums.items()}}
        history.append(rec)
        improved = val["mse"] < best_val
        if improved:
            best_val, best_epoch, bad = val["mse"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"ep {epoch:03d} | {epoch_times[-1]:.1f}s | train mse {rec['train_mse']:.4f} | "
              f"val mse {val['mse']:.4f} mae {val['mae']:.4f}{' *' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"early stop at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(best_state)
    if args.save_ckpt:
        torch.save(best_state, ckpt_path)
    peak_mem = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
    val, _, _ = evaluate(model, val_loader, device, args, series)
    test, pred, true = evaluate(model, test_loader, device, args, series, timing=True)
    if args.save_preds:
        np.savez_compressed(os.path.join(out_dir, "test_preds.npz"), pred=pred.astype(np.float32), true=true.astype(np.float32))

    cfg = {k: v for k, v in vars(args).items()}
    results = dict(config=cfg, git=git_rev(), n_params=n_params, best_epoch=best_epoch,
                   epochs_run=len(history), train_s_per_epoch=float(np.mean(epoch_times)),
                   peak_mem_mib=peak_mem, val=val, test=test, history=history)
    decomp = getattr(model, "decomp", None)
    if decomp is not None and decomp.mode == "gaussian":
        results["learned_sigma"] = float(decomp.sigma.detach())
    with open(res_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"TEST {args.dataset} H={args.pred_len} seed={args.seed} [{args.tag}] "
          f"mse {test['mse']:.4f} mae {test['mae']:.4f} (best epoch {best_epoch})", flush=True)
    return results


if __name__ == "__main__":
    main()
