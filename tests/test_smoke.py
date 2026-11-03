"""CPU smoke tests on a synthetic dataset: every ablation switch must build, train and evaluate."""
import json
import os
import subprocess
import sys

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import train  # noqa: E402
from pact.data import build_datasets, load_series, split_borders  # noqa: E402

N_VARS, N_ROWS, L = 3, 400, 32


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    t = np.arange(N_ROWS)
    rng = np.random.default_rng(0)
    vals = {f"v{j}": np.sin(2 * np.pi * t / (12 + 5 * j)) * 10 + 0.01 * t + rng.normal(0, 0.5, N_ROWS)
            for j in range(N_VARS - 1)}
    vals["OT"] = np.cos(2 * np.pi * t / 24) * 5 + rng.normal(0, 0.5, N_ROWS)
    start = datetime(2020, 1, 1)
    with open(root / "toy.csv", "w") as f:
        f.write("date," + ",".join(vals) + "\n")
        for i in range(N_ROWS):
            f.write(str(start + timedelta(hours=i)) + "," + ",".join(f"{vals[k][i]:.6f}" for k in vals) + "\n")
    # fake embedding store [T-L+1, N, d]
    emb_dir = root / "emb" / "Toy"
    emb_dir.mkdir(parents=True)
    np.save(emb_dir / f"fake_L{L}.npy", rng.normal(size=(N_ROWS - L + 1, N_VARS, 48)).astype(np.float16))
    return root


def base_args(root, tmp, **kw):
    args = dict(dataset="Toy", root_path=str(root), data_file="toy.csv", seq_len=L, pred_len=8,
                emb_root=str(root / "emb"), emb_tag="fake", d_model=16, n_heads=2, d_ff=32, patch_len=8, stride=4,
                n_prototypes=4, adapter_dim=8, adapter_heads=2, kernel_size=5, batch_size=16, epochs=2, patience=2,
                num_workers=0, device="cpu", out_root=str(tmp), tag="t")
    args.update(kw)
    argv = []
    for k, v in args.items():
        argv += [f"--{k}", str(v)]
    return argv


def test_splits_match_reference():
    b1, b2 = split_borders("custom", 1000, 96)
    assert b2 == [700, 800, 1000] and b1 == [0, 604, 704]
    b1, b2 = split_borders("ett_hour", 17420, 96)
    assert b2 == [8640, 11520, 14400] and b1 == [0, 8544, 11424]


def test_windows_cover_split(data_root):
    s = load_series("Toy", str(data_root), L, "toy.csv")
    tr, va, te = build_datasets(s, L, 8)
    b1, b2 = s["border1s"], s["border2s"]
    assert len(te) == b2[2] - b1[2] - L - 8 + 1
    x, y, start, _, xm = te[len(te) - 1]
    assert xm.shape == (L, 4) and float(xm.abs().max()) <= 0.5
    assert start + L + 8 == b2[2]  # the last test window ends exactly at the end of the series
    np.testing.assert_allclose(y.numpy(), s["scaled"][start + L:start + L + 8])


VARIANTS = {
    "full_patch": {},
    "full_variate": dict(token_mode="variate"),
    "no_decomp": dict(decomp="none"),
    "moving_avg": dict(decomp="moving_avg"),
    "no_pdpl": dict(pdpl=0),
    "vanilla_xattn": dict(bdca_iters=0),
    "bdca2": dict(bdca_iters=2),
    "no_bdca": dict(bdca_iters=-1),
    "llm_none": dict(llm="none"),
    "llm_learned": dict(llm="learned"),
    "llm_random": dict(llm="random"),
    "llm_shuffled": dict(llm="shuffled"),
    "nothing": dict(llm="none", pdpl=0, decomp="none"),
    "fusion_sum": dict(fusion="sum"),
    "timecma": dict(model="timecma", d_model=8),
    "v5": dict(model="v5", d_model=16, kernel_size=25),
    "long_context": dict(seq_len=64, prompt_len=L),
    "rlinear": dict(model="rlinear"),
    "patchtst": dict(model="patchtst", label_len=8),
    "itransformer": dict(model="itransformer", label_len=8),
    "dlinear": dict(model="dlinear", label_len=8),
    "timesnet": dict(model="timesnet", label_len=8, sched="type1"),
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_variant_trains(data_root, tmp_path, name):
    res = train.main(base_args(data_root, tmp_path, tag=name, **VARIANTS[name]))
    assert np.isfinite(res["test"]["mse"]) and res["test"]["n_windows"] > 0
    seq_len = VARIANTS[name].get("seq_len", L)
    assert os.path.exists(os.path.join(tmp_path, "Toy", name, f"L{seq_len}_H8_s2024", "results.json"))


def test_seed_reproducible(data_root, tmp_path):
    r1 = train.main(base_args(data_root, tmp_path / "a"))
    r2 = train.main(base_args(data_root, tmp_path / "b"))
    assert abs(r1["test"]["mse"] - r2["test"]["mse"]) < 1e-6


def test_gaussian_kernel_normalised():
    from pact.model import SeriesDecomp
    d = SeriesDecomp(11, "gaussian")
    w = d.kernel()
    assert torch.isclose(w.sum(), torch.tensor(1.0)) and torch.argmax(w) == 5
    x = torch.randn(2, 40, 3)
    tr, se = d(x)
    torch.testing.assert_close(tr + se, x)


def test_build_embeddings_tiny(data_root, tmp_path):
    """Random-init GPT-2 end to end; needs the gpt2 tokenizer (skipped offline)."""
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        pytest.skip("gpt2 tokenizer unavailable")
    cmd = [sys.executable, os.path.join(ROOT, "build_embeddings.py"), "--dataset", "Toy", "--root_path", str(data_root),
           "--data_file", "toy.csv", "--seq_len", str(L), "--random_init", "1", "--device", "cpu", "--fp16", "0",
           "--out_root", str(tmp_path), "--batch_size", "64", "--chunk", "100"]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True)
    arr = np.load(tmp_path / "Toy" / f"gpt2-randinit_L{L}.npy")
    assert arr.shape == (N_ROWS - L + 1, N_VARS, 768) and np.isfinite(arr).all() and np.abs(arr).sum(-1).min() > 0
    meta = json.load(open(tmp_path / "Toy" / f"gpt2-randinit_L{L}.json"))
    assert meta["n_windows"] == N_ROWS - L + 1
    subprocess.run(cmd + ["--value_norm", "window"], check=True, cwd=ROOT, capture_output=True)
    arr2 = np.load(tmp_path / "Toy" / f"gpt2-randinit-norm_L{L}.npy")
    assert arr2.shape == arr.shape and not np.allclose(arr, arr2)


def test_prompt_offset(data_root):
    """With prompt_len < seq_len the embedding row is the window of the last prompt_len steps."""
    from pact.data import EmbeddingStore
    s = load_series("Toy", str(data_root), 64, "toy.csv")
    store = EmbeddingStore(str(data_root / "emb" / "Toy" / f"fake_L{L}.npy"))
    tr, _, te = build_datasets(s, 64, 8, store, prompt_len=L)
    _, _, start, emb, _ = te[3]
    np.testing.assert_allclose(emb.numpy(), store[start + 64 - L])
