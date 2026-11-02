# PACT — clean re-implementation for the Pattern Recognition revision (PR-D-26-11306)

Prototype Alignment and Co-attention for Time series: Gaussian trend/seasonal decomposition, shared
per-variate encoder, prototype-based dynamic prompts (PDPL) gated by a frozen-LLM anchor, and
bi-directional co-attention (BDCA). TimeCMA is re-implemented in the same pipeline as the main baseline.

Everything reported in the revised manuscript must be produced by this repository and traceable to a
`results/<dataset>/<tag>/L<L>_H<H>_s<seed>/results.json`.

## Evaluation protocol

| Item | Setting |
|---|---|
| Splits | ETT 12/4/4 months; ECL/Weather/ILI/Traffic chronological 7:1:2 (train:val:test) |
| Scaling | z-score fitted on the training split; RevIN (non-affine) inside the model |
| Look-back / horizons | L=96, H ∈ {96,192,336,720}; ILI L=36, H ∈ {24,36,48,60} (same as TimeCMA) |
| Model selection | best **validation** MSE, early stopping (patience 10); test evaluated once |
| Loaders | train shuffled; val/test keep every window (`drop_last=False`) |
| Metrics | MSE/MAE averaged over all test windows, horizons and variates, on the z-scored scale (LTSF convention); raw-scale values are stored as `mse_raw`/`mae_raw` |
| Seeds | 2024, 2025, 2026 (mean ± sample std) |

## Workflow

```bash
uv venv && uv pip install torch numpy pandas transformers pytest   # or conda
python -m pytest -q tests                                          # CPU smoke tests (~1 min)

# 1. frozen-LLM prompt embeddings, one store per (dataset, L); independent of the horizon
python build_embeddings.py --dataset ETTh1 --root_path ./data --seq_len 96
python build_embeddings.py --dataset ETTh1 --root_path ./data --seq_len 96 --random_init 1   # control
#    large datasets over several GPUs:  --num_shards 4 --shard i  (one process per GPU), then --merge 1

# 2. experiments: generate a job file, run it over the GPUs of a machine
python tools/make_jobs.py --group pilot --extra "--root_path ./data" > jobs/pilot.txt
python tools/queue.py jobs/pilot.txt --gpus 0,1,2,3 --per_gpu 2          # --part i/k to split over machines

# 3. tables
python tools/aggregate.py --root results --tags pact,timecma
```

Groups in `tools/make_jobs.py`: `pilot` (token layout × learning rate), `main`, `timecma` (official
per-horizon hyper-parameters), `ablation`, `llm` (pre-trained-knowledge controls), `kproto`, `kernel`.

## Switches (train.py)

| Flag | Meaning |
|---|---|
| `--token_mode patch` | tokens = patches of each variate; PDPL/BDCA attend along time per variate |
| `--token_mode variate` | tokens = variates (whole window embedded); attention across variates |
| `--decomp gaussian / gaussian_fixed / moving_avg / none` | learnable-σ Gaussian (Eq. 1), fixed σ, Autoformer moving average, no decomposition |
| `--pdpl 0` | prompt = LLM anchor only (no prototype retrieval) |
| `--bdca_iters I` | I bi-directional rounds + final forward retrieval; `0` = single cross-attention; `-1` = no fusion module |
| `--llm stored` | pre-computed embeddings (`--emb_tag gpt2`, `gpt2-randinit`, other LLMs) |
| `--llm shuffled` | real embeddings of a *different* window (breaks sample alignment) |
| `--llm learned` | learnable vectors of the same shape through the same adapter (capacity-matched) |
| `--llm random` | fixed random vectors through the same adapter |
| `--llm none` | no LLM anchor (PDPL prompt only) |
| `--lambda_freq / --lambda_align / --lambda_div` | loss weights (0 disables the term) |
| `--model timecma` | TimeCMA baseline on the same data, embeddings and protocol |

## Differences from the code used for the first submission

The original experiments ran on a TimeCMA fork whose training scripts selected checkpoints on the test
split, dropped the last test batch, contained a CNN "detail refinement" branch not described in the
paper, a decomposition that behaved differently in training and inference, and attention across
variates although the paper described per-variate temporal attention. This repository implements the
method as described (with the `variate` layout kept as an explicit option) and fixes the protocol above.
