"""Measure where bitter input overrides sweet input.

LB1a drives proboscis extension. LB3a suppresses it. This script holds the
LB1a rate fixed and raises the LB3a rate in steps, then records MN9 spikes
at each step. The result is the acceptance curve of the wiring.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
SWEET_RATE = "200"
BITTER_RATES = ["0", "10", "25", "50", "75", "100", "150", "200"]


def ids_for(ann: pd.DataFrame, cell_type: str) -> list[int]:
    return [int(b) for b in ann[ann["type"] == cell_type]["bodyId"]]


def mn9_spikes(label: str, mn9: set[int]) -> int:
    files = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
    if not files:
        return -1
    sp = pd.read_parquet(files[0])
    return int(sp["flywire_id"].astype("int64").isin(mn9).sum())


def main() -> None:
    ann = pd.read_feather(ANN)
    mn9 = set(ids_for(ann, "MN9"))

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_NEU_EXC"] = ",".join(str(n) for n in ids_for(ann, "LB1a"))
    env["FLY_NEU_EXC2"] = ",".join(str(n) for n in ids_for(ann, "LB3a"))
    env["FLY_STIM_RATE"] = SWEET_RATE

    rows = []
    for rate in BITTER_RATES:
        label = f"dose_{rate}"
        env["FLY_STIM_RATE2"] = rate
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", "1", "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        count = mn9_spikes(label, mn9)
        rows.append({"bitter_hz": int(rate), "mn9": count})
        print(f"bitter {rate:>4s} Hz   MN9={count}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/dose_response.csv", index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
