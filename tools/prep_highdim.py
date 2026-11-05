"""Convert the Solar-Energy and PEMS08 benchmarks (as used by iTransformer) to the csv layout of pact.data.

Solar: solar_AL.txt (52,560 x 137, 10-minute PV output of Alabama plants, 2006; LSTNet release).
PEMS08: PEMS08.npz (17,856 x 170 x 3; flow, occupancy, speed); the traffic flow channel is used, as in iTransformer.
Neither file has timestamps; they are reconstructed from the documented start and sampling period (used only in
the prompt text and the time features).
  python tools/prep_highdim.py --solar solar_AL.txt --pems PEMS08.npz --out data
"""
import argparse
import os
from datetime import datetime, timedelta

import numpy as np


def write(path, values, start, step_min):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = datetime.fromisoformat(start)
    with open(path, "w") as f:
        f.write("date," + ",".join(str(i) for i in range(values.shape[1])) + "\n")
        for k, row in enumerate(values):
            f.write((t0 + timedelta(minutes=step_min * k)).strftime("%Y-%m-%d %H:%M:%S") + ","
                    + ",".join(f"{v:g}" for v in row) + "\n")
    print(path, values.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solar", required=True)
    p.add_argument("--pems", required=True)
    p.add_argument("--out", default="data")
    a = p.parse_args()
    write(os.path.join(a.out, "solar", "solar_AL.csv"), np.loadtxt(a.solar, delimiter=","), "2006-01-01 00:00:00", 10)
    write(os.path.join(a.out, "PEMS", "PEMS08.csv"), np.load(a.pems)["data"][:, :, 0], "2016-07-01 00:00:00", 5)


if __name__ == "__main__":
    main()
