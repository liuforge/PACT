"""Anchor embeddings from a frozen time-series foundation model (Sundial) instead of a language model.

For each look-back window s and variate j, the window (z-scored inside the window, i.e. scale-free like the
standardised prompts) is encoded by the frozen Sundial backbone and the hidden state of its last patch token is
stored. Output layout is identical to build_embeddings.py: {out_root}/{dataset}/{tag}_L{seq_len}.npy, float16,
[T - L + 1, N, d]. Sundial ships its own modelling code (trust_remote_code).

  python build_ts_embeddings.py --dataset ETTh1 --model hf_models/sundial-base-128m
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from numpy.lib.format import open_memmap

from pact.data import load_series


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--root_path", default="data")
    p.add_argument("--seq_len", type=int, default=96)
    p.add_argument("--model", default="hf_models/sundial-base-128m")
    p.add_argument("--tag", default="sundial-base")
    p.add_argument("--batch_size", type=int, default=4096, help="series per forward pass")
    p.add_argument("--out_root", default="embeddings")
    a = p.parse_args()
    from transformers import AutoModelForCausalLM
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=torch.float32)
    backbone = model.model.to(dev).eval()
    s = load_series(a.dataset, a.root_path, a.seq_len)
    x = s["scaled"]
    T, N = x.shape
    L = a.seq_len
    n_win = T - L + 1
    d = backbone.config.hidden_size
    os.makedirs(os.path.join(a.out_root, a.dataset), exist_ok=True)
    path = os.path.join(a.out_root, a.dataset, f"{a.tag}_L{L}.npy")
    out = open_memmap(path, mode="w+", dtype=np.float16, shape=(n_win, N, d))
    xt = torch.as_tensor(x, dtype=torch.float32)
    per = max(1, a.batch_size // N)  # windows per batch
    t0 = time.time()
    with torch.no_grad():
        for w0 in range(0, n_win, per):
            w1 = min(n_win, w0 + per)
            win = xt.unfold(0, L, 1)[w0:w1]  # [w, N, L]
            series = win.reshape(-1, L)
            mu = series.mean(-1, keepdim=True)
            sd = series.std(-1, keepdim=True, unbiased=False)
            sd = torch.where(sd > 1e-2, sd, torch.ones_like(sd))  # as Sundial's own RevIN
            series = ((series - mu) / sd).to(dev)
            h = backbone(input_ids=series, use_cache=False).last_hidden_state[:, -1]  # [w*N, d]
            out[w0:w1] = h.reshape(w1 - w0, N, d).float().cpu().numpy().astype(np.float16)
            if (w0 // per) % 20 == 0:
                print(f"[{a.dataset}] {w1}/{n_win} windows ({time.time() - t0:.0f}s)", flush=True)
    out.flush()
    json.dump(dict(dataset=a.dataset, seq_len=L, model=a.model, kind="time-series foundation model, last patch token",
                   window_norm=True, d_llm=d, n_windows=n_win, num_vars=N, dtype="float16"),
              open(path.replace(".npy", ".json"), "w"), indent=1)
    print(f"done -> {path} {out.shape}")


if __name__ == "__main__":
    main()
