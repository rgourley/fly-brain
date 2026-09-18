"""Test whether the MN9 latch fires with a probability that depends on input.

MN9 is bistable: it either latches near 125 spikes or stays near zero.
A single trial therefore measures nothing. This runs 16 trials per
condition and reports how often the latch sets.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
CONDITIONS = [("200", "0"), ("200", "50"), ("200", "100"),
              ("200", "200"), ("400", "100"), ("400", "200")]
TRIALS = "32"


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
    for sugar, bitter in CONDITIONS:
        label = f"prob_s{sugar}_b{bitter}"
        env["FLY_STIM_RATE"] = sugar
        env["FLY_STIM_RATE2"] = bitter
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", TRIALS, "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
        n = int(TRIALS)
        if not f:
            rows.append({"sugar": int(sugar), "bitter": int(bitter), "p_fire": 0.0})
            print(f"sugar {sugar:>4s} bitter {bitter:>4s} -> silent", flush=True)
            continue
        sp = pd.read_parquet(f[0])
        sp["flywire_id"] = sp["flywire_id"].astype("int64")
        per = (sp[sp["flywire_id"].isin(mn9)].groupby("trial").size()
               .reindex(range(n), fill_value=0))
        fired = int((per > 50).sum())
        rows.append({"sugar": int(sugar), "bitter": int(bitter),
                     "fired": fired, "trials": n, "p_fire": round(fired / n, 3)})
        print(f"sugar {sugar:>4s} bitter {bitter:>4s} -> {fired}/{n} fired "
              f"(p={fired/n:.2f})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/fire_probability.csv", index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
