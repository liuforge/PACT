"""Zero-shot evaluation of time-series foundation models on exactly the test targets of train.py.

The forecast targets are the test windows of the L=96 protocol (ILI: L=36); the model may read a
longer context (--context) that ends at the same time step, which only uses data before the target.
Inputs are the z-scored series and metrics are computed on that scale, as in results.json of train.py.

  python eval_foundation.py --model toto2 --ckpt hf_models/Toto-2.0-22m --dataset ETTh1 --context 512
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pact.data import load_series  # noqa: E402

BASE_L = {"ILI": 36}
HORIZONS = {"ILI": [24, 36, 48, 60]}


class Toto2:
    def __init__(self, ckpt, device):
        # toto2 imports gluonts.torch (-> lightning) only for its GluonTS predictor wrapper; forecast() does not use it
        if "gluonts.torch" not in sys.modules:
            stub = types.ModuleType("gluonts.torch")
            stub.PyTorchPredictor = object
            sys.modules["gluonts.torch"] = stub
        from toto2 import Toto2Model
        self.model = Toto2Model.from_pretrained(ckpt).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, ctx, H):  # ctx [B, Lc, N] -> [B, H, N]
        target = torch.as_tensor(ctx, device=self.device).transpose(1, 2).contiguous()
        mask = torch.ones_like(target, dtype=torch.bool)
        pad = -target.shape[-1] % 32  # the context must be a whole number of patches: left-pad as missing values
        if pad:
            target = F.pad(target, (pad, 0))
            mask = F.pad(mask, (pad, 0), value=False)
        q = self.model.forecast({"target": target, "target_mask": mask,
                                 "series_ids": torch.zeros(target.shape[:2], dtype=torch.long, device=self.device)},
                                horizon=H, decode_block_size=768, has_missing_values=bool(pad))
        return q[4].transpose(1, 2).float().cpu().numpy()  # median quantile


class TimesFM3:
    znorm = False  # set by --tfm_znorm: the forecaster's own per-series z-normalisation option

    def __init__(self, ckpt, device):
        from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
        self.model = TimesFM3Forecaster.from_pretrained(ckpt, device=str(device))

    @torch.no_grad()
    def __call__(self, ctx, H):  # multivariate contexts [N, Lc] (variate attention)
        outs = self.model.predict_batch(contexts=[c.T for c in ctx], horizon=H, use_znorm=self.znorm)
        return np.stack([np.asarray(o.forecast)[..., :H].reshape(ctx.shape[2], H).T for o in outs])


class TimeMoE:
    """Univariate; per-series z-normalisation as in the Time-MoE reference code / TSLib wrapper.

    The remote generate() relies on the transformers-4 cache API, so decoding is re-implemented without a
    KV cache: each step uses the largest prediction head not exceeding the remaining horizon (as generate does).
    """

    def _generate(self, seq, H):
        steps = []
        done = 0
        while done < H:
            # the remote code reshapes input_ids in place, so pass a copy
            logits = self.model(input_ids=seq.clone(), max_horizon_length=H - done, use_cache=False).logits[:, -1, :]
            step = logits[:, :H - done].to(seq.dtype)
            steps.append(step)
            seq = torch.cat([seq, step], dim=1)
            done += step.shape[1]
        return torch.cat(steps, dim=1)

    def __init__(self, ckpt, device):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True).float().to(device).eval()
        self.device = device
        # transformers>=5 materialises non-persistent buffers from the meta device without re-running __init__,
        # leaving the rotary tables uninitialised (NaN outputs); rebuild them as the remote code does.
        for mod in self.model.modules():
            if hasattr(mod, "inv_freq") and hasattr(mod, "_set_cos_sin_cache"):
                mod.inv_freq = 1.0 / (mod.base ** (torch.arange(0, mod.dim, 2, device=device).float() / mod.dim))
                mod._set_cos_sin_cache(seq_len=mod.max_position_embeddings, device=device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, ctx, H):
        B, Lc, N = ctx.shape
        x = torch.as_tensor(ctx, device=self.device).transpose(1, 2).reshape(B * N, Lc)
        mu, sd = x.mean(-1, keepdim=True), x.std(-1, keepdim=True) + 1e-5
        out = self._generate((x - mu) / sd, H)
        return (out * sd + mu).reshape(B, N, H).transpose(1, 2).float().cpu().numpy()


MODELS = {"toto2": Toto2, "timesfm3": TimesFM3, "timemoe": TimeMoE}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root_path", default="./data")
    p.add_argument("--context", type=int, default=512)
    p.add_argument("--pred_lens", default=None, help="comma list; default = the dataset's four horizons")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_windows", type=int, default=0, help="debug: evaluate only the first n test windows")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_root", default="./results")
    p.add_argument("--overwrite", type=int, default=0)
    p.add_argument("--tfm_znorm", type=int, default=0, help="TimesFM-3: use_znorm=True in predict_batch")
    a = p.parse_args()
    TimesFM3.znorm = bool(a.tfm_znorm)
    tag = a.tag or (f"zs_{os.path.basename(a.ckpt.rstrip('/'))}" + ("-znorm" if a.tfm_znorm else ""))
    device = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    L = BASE_L.get(a.dataset, 96)
    horizons = [int(h) for h in a.pred_lens.split(",")] if a.pred_lens else HORIZONS.get(a.dataset, [96, 192, 336, 720])
    series = load_series(a.dataset, a.root_path, L)
    data = series["scaled"]
    b1, b2 = series["border1s"][2], series["border2s"][2]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = MODELS[a.model](a.ckpt, device)
    net = getattr(model.model, "model", model.model)
    n_params = sum(p.numel() for p in net.parameters()) if isinstance(net, torch.nn.Module) else None

    for H in horizons:
        out_dir = os.path.join(a.out_root, a.dataset, f"{tag}_ctx{a.context}", f"L{L}_H{H}_s0")
        res_path = os.path.join(out_dir, "results.json")
        if os.path.exists(res_path) and not a.overwrite:
            print(f"[skip] {res_path}")
            continue
        targets = np.arange(b1 + L, b2 - H + 1)  # identical to the test windows of train.py
        if a.max_windows:
            targets = targets[:a.max_windows]
        if targets[0] - a.context < 0:
            raise ValueError("context longer than the available history")
        se, ae, n, times = 0.0, 0.0, 0, []
        for i in range(0, len(targets), a.batch_size):
            t = targets[i:i + a.batch_size]
            ctx = np.stack([data[s - a.context:s] for s in t]).astype(np.float32)
            y = np.stack([data[s:s + H] for s in t])
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = model(ctx, H)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            se += float(((pred - y) ** 2).sum())
            ae += float(np.abs(pred - y).sum())
            n += y.size
        test = dict(mse=se / n, mae=ae / n, n_windows=int(len(targets)),
                    infer_s_per_batch=float(np.mean(times[1:] if len(times) > 1 else times)))
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None
        os.makedirs(out_dir, exist_ok=True)
        cfg = dict(vars(a), tag=f"{tag}_ctx{a.context}", seq_len=L, pred_len=H, seed=0, model=a.model)
        json.dump(dict(config=cfg, n_params=n_params, zero_shot=True, train_s_per_epoch=0.0, peak_mem_mib=peak,
                       val=dict(mse=float("nan"), mae=float("nan")), test=test), open(res_path, "w"), indent=1)
        print(f"ZS {a.dataset} H={H} [{tag} ctx{a.context}] mse {test['mse']:.4f} mae {test['mae']:.4f} "
              f"({test['n_windows']} windows, {test['infer_s_per_batch']:.3f}s/batch)", flush=True)


if __name__ == "__main__":
    main()
