"""Pre-compute frozen-LLM prompt embeddings for every (window, variate) of a dataset.

For each look-back window starting at global row s and each variate j, a TimeCMA-style prompt
    "From [t1] to [t2], the values were v1, ..., vL every hour. The total trend value was T"
is built from the *raw* values and fed once through the frozen LLM; the hidden state of the last
prompt token is stored. Output: {out_root}/{dataset}/{tag}_L{seq_len}.npy, float16, shape
[T - L + 1, N, d_llm], plus a .json with the settings. The store does not depend on the horizon.

Large datasets can be split over GPUs with --num_shards/--shard (each shard writes a part file)
and merged with --merge. Progress is checkpointed, so an interrupted shard resumes where it stopped.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from numpy.lib.format import open_memmap

from pact.data import DATASETS, load_series
from pact.semantics import variable_sentence

TEMPLATE = "From {t1} to {t2}, the values were {values} {freq}. The total trend value was {trend}"
# --value_norm window: each prompt shows the window z-scored per variate (scale-free, robust to level shifts)
TEMPLATE_NORM = "From {t1} to {t2}, the standardized values were {values} {freq}. The total trend value was {trend}"


def get_args():
    p = argparse.ArgumentParser()
    a = p.add_argument
    a("--dataset", required=True)
    a("--root_path", default="./data")
    a("--data_file", default=None)
    a("--seq_len", type=int, default=96)
    a("--model", default="gpt2", help="HuggingFace model id or local path (causal LM)")
    a("--random_init", type=int, default=0, help="use the architecture with random weights (control for pre-training)")
    a("--tag", default=None, help="store name; default derived from --model (+ '-randinit')")
    a("--value_format", default=None, help="'int' (TimeCMA) or a format spec such as '.1f' (default: int / .1f)")
    a("--value_norm", default="none", choices=["none", "window", "global", "center"],
      help="none: raw values (TimeCMA); window: z-score each variate within the look-back window; global: z-score "
           "with the training-split mean/std (keeps the window's level, removes units); center: subtract the "
           "window mean (removes the level, keeps units)")
    a("--prompt", default="values", choices=["values", "variable", "variable_values", "text", "values_text"],
      help="values: window prompt only; variable: static description of the dataset and the variable "
           "(one embedding per variate, stored as <tag>_static.npy); variable_values: description + window prompt; "
           "text: the facts available before the forecast start (<root_path>/<dataset>.text.json, see "
           "tools/prep_mmd.py) instead of numbers; values_text: window prompt followed by those facts")
    a("--batch_size", type=int, default=256, help="prompts per forward pass")
    a("--chunk", type=int, default=512, help="windows per checkpointed chunk")
    a("--fp16", type=int, default=1)
    a("--device", default="cuda")
    a("--out_root", default="./embeddings")
    a("--num_shards", type=int, default=1)
    a("--shard", type=int, default=0)
    a("--merge", type=int, default=0)
    a("--seed", type=int, default=0)
    return p.parse_args()


def fmt_value(v, spec):
    return str(int(v)) if spec == "int" else format(float(v), spec)


def date_str(ts, dataset):
    if ts is None:
        return ""
    if dataset == "ILI":
        return ts.strftime("%d/%m/%Y")
    if dataset in ("ETTh1", "ETTh2", "ECL", "Traffic"):
        return ts.strftime("%d/%m/%Y %H:00")
    return ts.strftime("%d/%m/%Y %H:%M")


def load_llm(args, device):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    if args.random_init:
        torch.manual_seed(args.seed)
        model = AutoModel.from_config(AutoConfig.from_pretrained(args.model))
    else:
        model = AutoModel.from_pretrained(args.model)
    model = model.to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


@torch.no_grad()
def embed(prompts, tok, model, device, max_len):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    hidden = model(**enc).last_hidden_state  # [b, t, d]
    lengths = enc["attention_mask"].sum(1)
    last = lengths - 1  # right padding: index of the last real token
    n_trunc = int((lengths >= max_len).sum())
    return hidden[torch.arange(hidden.shape[0], device=device), last].float().cpu().numpy(), n_trunc


def paths(args):
    tag = args.tag or (os.path.basename(args.model.rstrip("/")) + ("-randinit" if args.random_init else "")
                       + {"values": "", "variable": "-var", "variable_values": "-var", "text": "-text",
                          "values_text": "-vtext"}[args.prompt]
                       + ({"window": "-norm", "global": "-gz", "center": "-ctr"}.get(args.value_norm, "")
                          if args.prompt not in ("variable", "text") else ""))
    d = os.path.join(args.out_root, args.dataset)
    os.makedirs(d, exist_ok=True)
    final = os.path.join(d, f"{tag}_L{args.seq_len}.npy")
    return tag, final


def shard_range(n, k, i):
    edges = np.linspace(0, n, k + 1).astype(int)
    return int(edges[i]), int(edges[i + 1])


def merge(args):
    tag, final = paths(args)
    parts = [final.replace(".npy", f".part{i}of{args.num_shards}.npy") for i in range(args.num_shards)]
    for p in parts:
        prog = json.load(open(p + ".progress"))
        if not prog["finished"]:
            raise RuntimeError(f"{p} is incomplete")
    arrs = [np.load(p, mmap_mode="r") for p in parts]
    total = sum(a.shape[0] for a in arrs)
    out = open_memmap(final, mode="w+", dtype=np.float16, shape=(total,) + arrs[0].shape[1:])
    o = 0
    for a in arrs:
        out[o:o + a.shape[0]] = a
        o += a.shape[0]
    out.flush()
    meta = json.load(open(parts[0] + ".meta.json"))
    meta.update(shape=list(out.shape), shards=args.num_shards)
    json.dump(meta, open(final.replace(".npy", ".json"), "w"), indent=1)
    for p in parts:
        for f in (p, p + ".progress", p + ".meta.json"):
            os.remove(f)
    print(f"merged -> {final} {out.shape}")


def main():
    args = get_args()
    if args.merge:
        return merge(args)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    series = load_series(args.dataset, args.root_path, args.seq_len, args.data_file)
    raw, dates = series["raw"], series["dates"]
    T, N = raw.shape
    L = args.seq_len
    n_windows = T - L + 1
    freq = DATASETS.get(args.dataset, (None, None, "every step"))[2]
    tag, final = paths(args)
    lo, hi = shard_range(n_windows, args.num_shards, args.shard)
    out_path = final if args.num_shards == 1 else final.replace(".npy", f".part{args.shard}of{args.num_shards}.npy")
    prog_path = out_path + ".progress"

    if args.value_format is None:
        args.value_format = "int" if args.value_norm == "none" else ".1f"
    template = TEMPLATE_NORM if args.value_norm in ("window", "global") else TEMPLATE
    tok, model = load_llm(args, device)
    d_llm = model.config.hidden_size
    var_text = [variable_sentence(args.dataset, c, j) for j, c in enumerate(series["columns"])]
    texts = None
    if args.prompt in ("text", "values_text"):  # texts[t]: facts dated before row t (no look-ahead)
        texts = json.load(open(os.path.join(args.root_path, f"{args.dataset}.text.json")))
        assert len(texts) == T, (len(texts), T)
    if args.prompt == "variable":  # static: one prompt per variate
        vec, _ = embed(var_text, tok, model, device, 1024)
        static = final.replace(f"_L{L}.npy", "_static.npy")
        np.save(static, vec.astype(np.float16))
        json.dump(dict(dataset=args.dataset, model=args.model, random_init=args.random_init, prompts=var_text,
                       d_llm=d_llm, num_vars=N, dtype="float16"), open(static.replace(".npy", ".json"), "w"), indent=1)
        print(f"example: {var_text[-1]}\ndone -> {static} {vec.shape}")
        return
    max_len = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 2048)
    meta = dict(dataset=args.dataset, seq_len=L, model=args.model, random_init=args.random_init,
                value_format=args.value_format, value_norm=args.value_norm, template=template, freq=freq, d_llm=d_llm,
                n_windows=n_windows, num_vars=N, dtype="float16")

    if os.path.exists(prog_path):
        prog = json.load(open(prog_path))
        out = open_memmap(out_path, mode="r+")
    else:
        prog = dict(done=0, finished=False)
        out = open_memmap(out_path, mode="w+", dtype=np.float16, shape=(hi - lo, N, d_llm))
        json.dump(meta, open(out_path + ".meta.json" if args.num_shards > 1 else final.replace(".npy", ".json"), "w"), indent=1)
    if prog["finished"]:
        print("already finished")
        return

    # formatted value strings, column by column (reused by every window); window-normalised prompts are per window
    if args.value_norm == "global":  # the training-split z-scoring used by the models, applied to the whole series
        gz = series["scaled"]
        col_strs = [[fmt_value(v, args.value_format) for v in gz[:, j]] for j in range(N)]
    elif args.value_norm == "none":
        col_strs = [[fmt_value(v, args.value_format) for v in raw[:, j]] for j in range(N)]
    else:
        col_strs = None
    date_strs = [date_str(ts, args.dataset) for ts in dates] if dates is not None else [""] * T
    t_start = time.time()
    n_trunc = 0
    s = lo + prog["done"]
    while s < hi:
        e = min(hi, s + args.chunk)
        prompts, slots = [], []
        for w in range(s, e):
            if col_strs is None:
                win = raw[w:w + L]
                z = win - win.mean(0) if args.value_norm == "center" else (win - win.mean(0)) / (win.std(0) + 1e-5)
            for j in range(N):
                if args.value_norm == "global":
                    values = ", ".join(col_strs[j][w:w + L])
                    trend = f"{float(gz[w + L - 1, j] - gz[w, j]):.2f}"
                elif args.value_norm == "center":
                    values = ", ".join(fmt_value(v, args.value_format) for v in z[:, j])
                    trend = f"{float(raw[w + L - 1, j] - raw[w, j]):f}"
                elif col_strs is None:
                    values = ", ".join(fmt_value(v, args.value_format) for v in z[:, j])
                    trend = f"{float(z[-1, j] - z[0, j]):.2f}"
                else:
                    values = ", ".join(col_strs[j][w:w + L])
                    trend = f"{float(raw[w + L - 1, j] - raw[w, j]):f}"
                p = template.format(t1=date_strs[w], t2=date_strs[w + L - 1], values=values, freq=freq, trend=trend)
                if texts is not None:  # window s..s+L-1 may use the facts available when row s+L starts
                    ctx = texts[min(w + L, T - 1)] or "No relevant reports."
                    p = ctx if args.prompt == "text" else p + ". " + ctx
                prompts.append(var_text[j] + " " + p if args.prompt == "variable_values" else p)
                slots.append((w - lo, j))
        if s == lo + prog["done"] and prog["done"] == 0:
            n_tok = len(tok(prompts[0])["input_ids"])
            print(f"example prompt ({n_tok} tokens): {prompts[0][:300]}...")
        for b in range(0, len(prompts), args.batch_size):
            batch = prompts[b:b + args.batch_size]
            vec, k = embed(batch, tok, model, device, max_len)
            n_trunc += k
            for (r, j), v in zip(slots[b:b + args.batch_size], vec):
                out[r, j] = v
        out.flush()
        prog["done"] = e - lo
        json.dump(prog, open(prog_path, "w"))
        rate = (e - lo - (s - lo)) * N / max(1e-6, time.time() - t_start)
        print(f"[{args.dataset} shard {args.shard}] {e - lo}/{hi - lo} windows | {rate:.0f} prompts/s", flush=True)
        t_start = time.time()
        s = e
    prog["finished"] = True
    json.dump(prog, open(prog_path, "w"))
    if n_trunc:
        print(f"warning: {n_trunc} prompts truncated to {max_len} tokens")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
