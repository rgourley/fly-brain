"""Test whether lateralized odor input moves the steering readout.

Stimulate the left olfactory channels alone, then the right alone, at two
drive levels. If DNa02 tracks the input, the ipsilateral cell should lead
and the lead should swap when the stimulated side swaps.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
CHANNELS = ["ORN_DM1", "ORN_DM2", "ORN_VA2", "ORN_VM5d"]
RATES = ["200", "400"]


def matched_sides(ann: pd.DataFrame) -> tuple[list[int], list[int]]:
    left, right = [], []
    for ch in CHANNELS:
        rows = ann[ann["type"] == ch]
        l = sorted(int(b) for b in rows[rows["instance"] == f"{ch}_L"]["bodyId"])
        r = sorted(int(b) for b in rows[rows["instance"] == f"{ch}_R"]["bodyId"])
        n = min(len(l), len(r))
        left += l[:n]
        right += r[:n]
    return left, right


def main() -> None:
    ann = pd.read_feather(ANN)
    left, right = matched_sides(ann)
    dna = {s: int(ann[ann["instance"] == f"DNa02_{s}"]["bodyId"].iloc[0]) for s in "LR"}
    print(f"{len(left)} neurons per side, DNa02 {dna}")

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")

    rows = []
    for rate in RATES:
        for stim_side, neurons in (("L", left), ("R", right)):
            label = f"lat_{stim_side}_{rate}"
            env["FLY_NEU_EXC"] = ",".join(str(n) for n in neurons)
            env["FLY_STIM_RATE"] = rate
            subprocess.run(
                [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
                 "--t_run", "1", "--n_run", "2", "--run-label", label],
                cwd=ROOT, env=env, capture_output=True, check=True,
            )
            f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
            sp = pd.read_parquet(f[0])
            sp["flywire_id"] = sp["flywire_id"].astype("int64")
            n_trials = sp["trial"].nunique()
            l = int((sp["flywire_id"] == dna["L"]).sum()) / n_trials
            r = int((sp["flywire_id"] == dna["R"]).sum()) / n_trials
            rows.append({"stim": stim_side, "hz": int(rate),
                         "DNa02_L": l, "DNa02_R": r, "ipsi_minus_contra":
                         (l - r) if stim_side == "L" else (r - l)})
            print(f"stim {stim_side} @ {rate}Hz -> DNa02_L={l:.1f} DNa02_R={r:.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/lateral_test.csv", index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
