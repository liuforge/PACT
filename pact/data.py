"""Dataset loading with the standard LTSF splits.

Splits follow the convention shared by Informer/Autoformer/PatchTST/iTransformer/TimeCMA:
  * ETTh1/ETTh2: 12/4/4 months (hourly);  ETTm1/ETTm2: 12/4/4 months (15-min)
  * all other datasets: chronological 7:1:2 (train:val:test)
Values are z-scored with statistics fitted on the training portion only.

Each window is identified by its *global* start index ``s`` (x = rows s..s+L-1), which is
also the row index into the pre-computed LLM prompt-embedding store (see build_embeddings.py).
"""
import csv
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

# name -> (default csv path under root_path, split kind, sampling phrase used in the prompt)
DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", "ett_hour", "every hour"),
    "ETTh2": ("ETT-small/ETTh2.csv", "ett_hour", "every hour"),
    "ETTm1": ("ETT-small/ETTm1.csv", "ett_minute", "every 15 minutes"),
    "ETTm2": ("ETT-small/ETTm2.csv", "ett_minute", "every 15 minutes"),
    "ECL": ("electricity/electricity.csv", "custom", "every hour"),
    "Weather": ("weather/weather.csv", "custom", "every 10 minutes"),
    "ILI": ("illness/national_illness.csv", "custom", "every week"),
    "Traffic": ("traffic/traffic.csv", "custom", "every hour"),
}


def split_borders(kind, n_rows, seq_len):
    """Return (border1s, border2s) for train/val/test, identical to the reference code bases."""
    if kind in ("ett_hour", "ett_minute"):
        unit = 30 * 24 * (4 if kind == "ett_minute" else 1)
        n_train, n_val, n_test = 12 * unit, 4 * unit, 4 * unit
        border1s = [0, n_train - seq_len, n_train + n_val - seq_len]
        border2s = [n_train, n_train + n_val, n_train + n_val + n_test]
    else:
        n_train = int(n_rows * 0.7)
        n_test = int(n_rows * 0.2)
        n_val = n_rows - n_train - n_test
        border1s = [0, n_train - seq_len, n_rows - n_test - seq_len]
        border2s = [n_train, n_train + n_val, n_rows]
    return border1s, border2s


def resolve_csv(dataset, root_path, data_file=None):
    if data_file:
        return data_file if os.path.isabs(data_file) else os.path.join(root_path, data_file)
    rel = DATASETS[dataset][0]
    # registry layout (root/electricity/electricity.csv), flat layout (root/electricity.csv), TimeCMA layout (root/ECL.csv)
    candidates = [rel, os.path.basename(rel), f"{dataset}.csv"]
    for c in candidates:
        if os.path.exists(os.path.join(root_path, c)):
            return os.path.join(root_path, c)
    raise FileNotFoundError(f"{dataset}: none of {candidates} under {root_path}")


def load_series(dataset, root_path, seq_len, data_file=None, target="OT"):
    """Load a dataset csv and return raw values, z-scored values, dates and split borders."""
    kind = DATASETS[dataset][1] if dataset in DATASETS else "custom"
    with open(resolve_csv(dataset, root_path, data_file), newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r]
    has_date = header[0] == "date"
    cols = header[1:] if has_date else header[:]
    order = list(range(len(cols)))
    if kind == "custom" and target in cols:  # target last, as in the reference loaders
        t = cols.index(target)
        order = [i for i in order if i != t] + [t]
    off = 1 if has_date else 0
    raw = np.array([[float(r[off + i]) for i in order] for r in body], dtype=np.float32)
    cols = [cols[i] for i in order]
    dates = [datetime.fromisoformat(r[0]) for r in body] if has_date else None
    border1s, border2s = split_borders(kind, len(body), seq_len)
    train = raw[border1s[0]:border2s[0]]
    mean = train.mean(0)
    std = train.std(0)
    std[std == 0] = 1.0
    scaled = ((raw - mean) / std).astype(np.float32)
    return dict(raw=raw, scaled=scaled, dates=dates, columns=cols, mean=mean, std=std,
                border1s=border1s, border2s=border2s, kind=kind)


class EmbeddingStore:
    """Read-only view on a pre-computed [num_windows, N, d_llm] float16 .npy store.

    Opened lazily so that DataLoader workers each get their own memmap instead of a pickled copy.
    """

    def __init__(self, path, index_map=None):
        self.path = path
        meta_path = os.path.splitext(path)[0] + ".json"
        self.meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        self.index_map = index_map
        self._arr = None

    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="r")
        return self._arr

    @property
    def d_llm(self):
        return int(self.arr.shape[-1])

    def __len__(self):
        return int(self.arr.shape[0])

    def __getitem__(self, s):
        if self.index_map is not None:
            s = int(self.index_map[s])
        return np.asarray(self.arr[s], dtype=np.float32)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arr"] = None
        return state


class WindowDataset(Dataset):
    def __init__(self, scaled, border1, border2, seq_len, pred_len, emb_store=None):
        self.data = scaled
        self.start = border1
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = border2 - border1 - seq_len - pred_len + 1
        self.emb = emb_store
        if self.n <= 0:
            raise ValueError(f"split too short for seq_len={seq_len}, pred_len={pred_len}")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.start + i
        x = self.data[s:s + self.seq_len]
        y = self.data[s + self.seq_len:s + self.seq_len + self.pred_len]
        emb = self.emb[s] if self.emb is not None else np.zeros(0, dtype=np.float32)
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy()), s, torch.from_numpy(emb)


def build_datasets(series, seq_len, pred_len, emb_store=None):
    b1, b2 = series["border1s"], series["border2s"]
    return [WindowDataset(series["scaled"], b1[k], b2[k], seq_len, pred_len, emb_store) for k in range(3)]
