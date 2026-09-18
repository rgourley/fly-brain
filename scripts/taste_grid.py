"""Map MN9 response across a grid of sugar and bitter concentrations.

If the response surface is a plane, the taste circuit is computing a
weighted difference and a threshold, and the fly adds nothing a linear
model could not. Structure in the surface means the wiring is doing
something of its own.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
SUGAR = ["50", "100", "200", "400"]
BITTER = ["0", "25", "50", "100"]


def main() -> None:
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    ids = lambda t: ",".join(str(int(b)) for b in ann[ann["type"] == t]["bodyId"])
    mn9 = set(int(b) for b in ann[ann["type"] == "MN9"]["bodyId"])

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_NEU_EXC"] = ids("LB1a")
    env["FLY_NEU_EXC2"] = ids("LB3a")

    rows = []
    for s in SUGAR:
        for b in BITTER:
            label = f"grid_s{s}_b{b}"
            env["FLY_STIM_RATE"] = s
            env["FLY_STIM_RATE2"] = b
            subprocess.run(
                [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
                 "--t_run", "1", "--n_run", "1", "--run-label", label],
                cwd=ROOT, env=env, capture_output=True, check=True,
            )
            f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
            n = 0
            if f:
                sp = pd.read_parquet(f[0])
                n = int(sp["flywire_id"].astype("int64").isin(mn9).sum())
            rows.append({"sugar": int(s), "bitter": int(b), "mn9": n})
            print(f"sugar {s:>4s} bitter {b:>4s} -> MN9={n}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/taste_grid.csv", index=False)
    print()
    print(df.pivot(index="sugar", columns="bitter", values="mn9").to_string())


if __name__ == "__main__":
    main()
