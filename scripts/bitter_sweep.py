"""Find labellar bristle types that suppress proboscis extension.

LB1a drives MN9, so LB1a is a sweet channel. This script adds each other
labellar type alongside LB1a and measures MN9 again. A type that lowers
the MN9 count acts as a bitter channel.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
T_RUN = "1"


def ids_for(ann: pd.DataFrame, cell_type: str) -> list[int]:
    return [int(b) for b in ann[ann["type"] == cell_type]["bodyId"]]


def mn9_spikes(label: str, mn9: set[int]) -> int:
    path = ROOT / "data" / "results" / label / "round_01"
    files = sorted(path.glob("*.parquet"))
    if not files:
        return -1
    sp = pd.read_parquet(files[0])
    return int(sp["flywire_id"].astype("int64").isin(mn9).sum())


def main() -> None:
    ann = pd.read_feather(ANN)
    sweet = ids_for(ann, "LB1a")
    mn9 = set(ids_for(ann, "MN9"))

    types = [t for t in sorted(ann[(ann["class"] == "gustatory")
                                   & (ann["subclass"] == "labellar bristle")]["type"].dropna().unique())
             if t != "LB1a"]

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")

    rows = []
    runs = [("baseline", sweet)] + [(t, sweet + ids_for(ann, t)) for t in types]
    for name, neurons in runs:
        label = f"sweep_{name}"
        env["FLY_NEU_EXC"] = ",".join(str(n) for n in neurons)
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", T_RUN, "--n_run", "1", "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        count = mn9_spikes(label, mn9)
        rows.append({"type": name, "n_stim": len(neurons), "mn9": count})
        print(f"{name:10s} stim={len(neurons):3d}  MN9={count}", flush=True)

    df = pd.DataFrame(rows)
    base = int(df[df["type"] == "baseline"]["mn9"].iloc[0])
    df["delta"] = df["mn9"] - base
    out = ROOT / "data/results/bitter_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\nbaseline MN9 = {base}")
    print(df[df["type"] != "baseline"].sort_values("delta").to_string(index=False))


if __name__ == "__main__":
    main()
