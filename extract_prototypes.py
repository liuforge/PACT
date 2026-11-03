"""Project every PDPL prototype onto the training patches that retrieve it most strongly (ProtoPNet-style).

For a run trained with --save_ckpt 1, the model is re-run over the training windows; for each prototype k
the --top patches (of the chosen branch, after RevIN and decomposition) with the largest retrieval weight
A[., k] are kept. Output: an .npz with the patches, their weights, prototype usage and the retrieval
entropy, consumed by tools/plot_prototypes.py.

  python extract_prototypes.py --run results/ETTh1/viz/L96_H96_s2024 --root_path data --out viz/ETTh1.npz
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pact.data import EmbeddingStore, build_datasets, load_series  # noqa: E402
from pact.model import PACT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory with results.json and best.pt")
    p.add_argument("--root_path", default="data")
    p.add_argument("--branch", default="seasonal", choices=["seasonal", "trend", "main"])
    p.add_argument("--top", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    cfg = SimpleNamespace(**json.load(open(os.path.join(a.run, "results.json")))["config"])
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    series = load_series(cfg.dataset, a.root_path, cfg.seq_len, cfg.data_file)
    store = None
    if cfg.llm in ("stored", "shuffled"):
        name = f"{cfg.emb_tag}_static.npy" if getattr(cfg, "emb_static", 0) else f"{cfg.emb_tag}_L{cfg.prompt_len}.npy"
        store = EmbeddingStore(os.path.join(cfg.emb_root, cfg.dataset, name), static=bool(getattr(cfg, "emb_static", 0)))
    train_set, _, _ = build_datasets(series, cfg.seq_len, cfg.pred_len, store, getattr(cfg, "prompt_len", None))
    model = PACT(cfg).to(device)
    model.load_state_dict(torch.load(os.path.join(a.run, "best.pt"), map_location=device))
    model.eval()
    enc = model.encoder
    K = model.pdpl.prototypes.shape[0]
    best_w = torch.full((K, a.top), -1.0)
    best_p = torch.zeros((K, a.top, enc.patch_len))
    best_var = torch.zeros((K, a.top), dtype=torch.long)
    usage = torch.zeros(K)
    ent_sum, n_tok = 0.0, 0
    gen = torch.Generator().manual_seed(0)
    rand_p = []  # uniform sample of training patches: baseline for the coherence of retrieved patches
    loader = torch.utils.data.DataLoader(train_set, batch_size=a.batch_size, shuffle=False)
    with torch.no_grad():
        for x, _, _, emb, xm in loader:
            x = x.to(device)
            emb = emb.to(device) if emb.numel() else None
            _, _, det = model(x, emb, xm.to(device), return_details=True)
            attn = det[f"proto_attn_{a.branch}"]  # [B*N, T, K]
            z = model.revin.norm(x) if model.revin is not None else x
            parts = dict(zip(model.branches, model.decomp(z))) if model.decomp is not None else {"main": z}
            xb = parts[a.branch]
            B, L, N = xb.shape
            seq = xb.permute(0, 2, 1).reshape(B * N, 1, L)
            if enc.pad_end:
                seq = F.pad(seq, (0, enc.pad_end), mode="replicate")
            patches = seq.unfold(-1, enc.patch_len, enc.stride).squeeze(1)  # [B*N, T, patch_len]
            w = attn.reshape(-1, K).cpu()
            pt = patches.reshape(-1, enc.patch_len).cpu()
            var = torch.arange(B * N).repeat_interleave(attn.shape[1]) % N
            usage += w.sum(0)
            rand_p.append(pt[torch.randint(len(pt), (min(32, len(pt)),), generator=gen)])
            ent_sum += float(-(w * w.clamp_min(1e-9).log()).sum())
            n_tok += w.shape[0]
            for k in range(K):  # merge this batch's best patches for prototype k
                cand_w = torch.cat([best_w[k], w[:, k]])
                cand_p = torch.cat([best_p[k], pt])
                cand_v = torch.cat([best_var[k], var])
                idx = torch.topk(cand_w, a.top).indices
                best_w[k], best_p[k], best_var[k] = cand_w[idx], cand_p[idx], cand_v[idx]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    rand_p = torch.cat(rand_p)[:2000]
    np.savez_compressed(a.out, patches=best_p.numpy(), weights=best_w.numpy(), variates=best_var.numpy(),
                        random_patches=rand_p.numpy(),
                        usage=(usage / usage.sum()).numpy(), entropy=ent_sum / n_tok / np.log(K),
                        columns=np.array(series["columns"]), dataset=cfg.dataset, branch=a.branch,
                        patch_len=enc.patch_len, pdpl_mode=getattr(cfg, "pdpl_mode", "dot"))
    print(f"{cfg.dataset} {a.branch}: K={K}, normalised retrieval entropy {ent_sum / n_tok / np.log(K):.3f} -> {a.out}")


if __name__ == "__main__":
    main()
