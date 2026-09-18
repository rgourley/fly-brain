"""Measure how steering asymmetry scales with wind strength.

Wind in the world ranges from still to strong. This drives the left
wind_gravity population across a range of rates and records how far the
descending output leans, so a calm day and a gusty one can be told apart.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
RATES = ["0", "25", "50", "100", "200", "400"]


def main() -> None:
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    comp = set(pd.read_csv(ROOT / "data/malecns/2026_Completeness_malecns.csv",
                           index_col=0).index)
    w = ann[(ann["subclass"] == "wind_gravity") & (ann["bodyId"].isin(comp))].copy()
    w["side"] = w["instance"].str.extract(r"_(L|R)$")[0]
    left = [int(b) for b in w[w["side"] == "L"]["bodyId"]]

    dna = {s: int(ann[ann["instance"] == f"DNa02_{s}"]["bodyId"].iloc[0]) for s in "LR"}
    dn = ann[ann["superclass"] == "descending_neuron"].copy()
    dn["side"] = dn["instance"].str.extract(r"_(L|R)$")[0]
    dn_ids = {s: set(dn[dn["side"] == s]["bodyId"]) for s in "LR"}

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_NEU_EXC"] = ",".join(str(n) for n in left)

    rows = []
    for rate in RATES:
        label = f"windstr_{rate}"
        env["FLY_STIM_RATE"] = rate
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", "2", "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
        if not f:
            rows.append({"wind_hz": int(rate), "dna02_L": 0, "dna02_R": 0, "dn_lean_pct": 0.0})
            print(f"wind {rate:>4s} Hz -> silent", flush=True)
            continue
        sp = pd.read_parquet(f[0])
        sp["flywire_id"] = sp["flywire_id"].astype("int64")
        counts = sp["flywire_id"].value_counts() / sp["trial"].nunique()
        l = float(counts.reindex(list(dn_ids["L"])).fillna(0).sum())
        r = float(counts.reindex(list(dn_ids["R"])).fillna(0).sum())
        lean = (l - r) / max(l + r, 1) * 100
        rows.append({"wind_hz": int(rate),
                     "dna02_L": float(counts.get(dna["L"], 0)),
                     "dna02_R": float(counts.get(dna["R"], 0)),
                     "dn_lean_pct": round(lean, 1)})
        print(f"wind {rate:>4s} Hz -> DNa02 L={counts.get(dna['L'],0):.1f} "
              f"R={counts.get(dna['R'],0):.1f}  DN lean={lean:+.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/wind_strength.csv", index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
