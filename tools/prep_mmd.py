"""Convert the Time-MMD domains (as preprocessed by MM-TSFlib, with search facts already matched to each row) to
the csv layout of this code base, plus the text available before each row.

Writes <out>/MMD_<domain>.csv (date, OT; sorted by date, rows without a target dropped) and
<out>/MMD_<domain>.text.json: texts[t] = the facts retrieved for the k periods preceding row t (MM-TSFlib column
Final_Search_<k>). For a look-back window covering rows s..s+L-1, the forecaster may use texts[s+L]: every fact in
it is dated before row s+L starts, i.e. within the look-back period; this is checked here.

  python tools/prep_mmd.py --src ../data_ext/MM-TSFlib --out data --k 4
"""
import argparse
import csv
import glob
import json
import os
import re
from datetime import date

DOMAINS = {"Algriculture": "Agriculture", "Climate": "Climate", "Economy": "Economy", "Energy": "Energy",
           "Public_Health": "Health", "Security": "Security", "SocialGood": "SocialGood", "Traffic": "Traffic"}
FACT_DATE = re.compile(r"(\d{4}-\d{2}-\d{2}):")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--k", type=int, default=4, help="Final_Search_<k>: facts from the k preceding periods")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for folder, name in DOMAINS.items():
        f = glob.glob(os.path.join(a.src, folder, "*.csv"))[0]
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8", errors="ignore")) if r["OT"].strip()]
        rows.sort(key=lambda r: r["start_date"])
        texts = [(r[f"Final_Search_{a.k}"] or "").strip() for r in rows]
        late = 0
        for r, t in zip(rows, texts):  # no fact may be dated on or after the start of its row
            ds = [date.fromisoformat(d) for d in FACT_DATE.findall(t)]
            if ds and max(ds) >= date.fromisoformat(r["start_date"][:10]):
                late += 1
        with open(os.path.join(a.out, f"MMD_{name}.csv"), "w", newline="") as g:
            w = csv.writer(g)
            w.writerow(["date", "OT"])
            for r in rows:
                w.writerow([r["start_date"][:10], r["OT"]])
        json.dump(texts, open(os.path.join(a.out, f"MMD_{name}.text.json"), "w"))
        n_empty = sum(1 for t in texts if len(t) < 40)
        print(f"MMD_{name}: {len(rows)} rows {rows[0]['start_date'][:10]}..{rows[-1]['start_date'][:10]} | "
              f"rows with facts dated too late: {late} | (near-)empty texts: {n_empty}")


if __name__ == "__main__":
    main()
