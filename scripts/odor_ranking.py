"""Test whether odor identity changes how often the feeding neuron fires.

Wide olfactory input produces distinguishable brain states. If the feeding
latch fires at different rates for different odors, the fly can rank
inputs using an output that already means something in the animal.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
N_PATTERNS = 3
TRIALS = "4"


def main() -> None:
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    comp = set(pd.read_csv(ROOT / "data/malecns/2026_Completeness_malecns.csv",
                           index_col=0).index)
    orn = ann[(ann["class"] == "olfactory") & (ann["bodyId"].isin(comp))]
    channels = sorted(orn["type"].dropna().unique())
    by_channel = {c: [int(b) for b in orn[orn["type"] == c]["bodyId"]] for c in channels}
    mn9 = set(int(b) for b in ann[ann["type"] == "MN9"]["bodyId"])
    sugar = ",".join(str(int(b)) for b in ann[ann["type"] == "LB1a"]["bodyId"])

    rng = np.random.default_rng(7)
    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_NEU_EXC2"] = sugar
    env["FLY_STIM_RATE"] = os.environ.get("ODOR_HZ", "200")
    env["FLY_STIM_RATE2"] = "200"

    rows = []
    for p in range(N_PATTERNS):
        chosen = rng.choice(channels, size=len(channels) // 2, replace=False)
        neurons = [n for c in chosen for n in by_channel[c]]
        label = f"odorrank_{p}"
        env["FLY_NEU_EXC"] = ",".join(str(n) for n in neurons)
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", TRIALS, "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
        n = int(TRIALS)
        sp = pd.read_parquet(f[0])
        sp["flywire_id"] = sp["flywire_id"].astype("int64")
        per = (sp[sp["flywire_id"].isin(mn9)].groupby("trial").size()
               .reindex(range(n), fill_value=0))
        fired = int((per > 50).sum())
        rows.append({"odor": p, "stimulated": len(neurons),
                     "fired": fired, "trials": n, "p_fire": round(fired / n, 3)})
        print(f"odor {p} ({len(neurons)} neurons): {fired}/{n} fired "
              f"(p={fired/n:.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/odor_ranking.csv", index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
