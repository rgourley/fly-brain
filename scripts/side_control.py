"""Check that matched bilateral input produces a balanced steering command.

The annotation labels more olfactory neurons on the right than the left, so
stimulating every neuron in a channel would bias DNa02 toward one side. This
script stimulates an equal count per side and reports the DNa02 spike counts.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
CHANNELS = ["ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_VM5d"]


def matched_sides(ann: pd.DataFrame, channels: list[str]) -> tuple[list[int], list[int]]:
    left, right = [], []
    for ch in channels:
        rows = ann[ann["type"] == ch]
        l = sorted(int(b) for b in rows[rows["instance"] == f"{ch}_L"]["bodyId"])
        r = sorted(int(b) for b in rows[rows["instance"] == f"{ch}_R"]["bodyId"])
        n = min(len(l), len(r))
        left += l[:n]
        right += r[:n]
        print(f"{ch:10s} L={len(l):3d} R={len(r):3d} -> using {n} per side")
    return left, right


def main() -> None:
    ann = pd.read_feather(ANN)
    left, right = matched_sides(ann, CHANNELS)
    dna02 = {
        "L": int(ann[ann["instance"] == "DNa02_L"]["bodyId"].iloc[0]),
        "R": int(ann[ann["instance"] == "DNa02_R"]["bodyId"].iloc[0]),
    }

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_NEU_EXC"] = ",".join(str(n) for n in left + right)
    env["FLY_STIM_RATE"] = "200"

    label = "side_control"
    subprocess.run(
        [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
         "--t_run", "1", "--n_run", "4", "--run-label", label],
        cwd=ROOT, env=env, capture_output=True, check=True,
    )

    files = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
    sp = pd.read_parquet(files[0])
    sp["flywire_id"] = sp["flywire_id"].astype("int64")
    print()
    for trial, grp in sp.groupby("trial"):
        l = int((grp["flywire_id"] == dna02["L"]).sum())
        r = int((grp["flywire_id"] == dna02["R"]).sum())
        total = l + r
        bias = (r - l) / total * 100 if total else 0.0
        print(f"trial {trial}: DNa02 L={l:4d} R={r:4d}  bias={bias:+.1f}% toward R")


if __name__ == "__main__":
    main()
