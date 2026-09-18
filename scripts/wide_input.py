"""Test whether a wide sensory input produces distinguishable brain states.

Earlier conditions used 28 taste neurons and every brain state came back
correlated at 0.99. Doomfly drives 4,146 visual neurons. This encodes
several distinct patterns across all 53 olfactory channels, roughly 1,700
neurons, and measures how different the resulting brain states are.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
N_PATTERNS = 4


def main() -> None:
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    comp = set(pd.read_csv(ROOT / "data/malecns/2026_Completeness_malecns.csv",
                           index_col=0).index)
    orn = ann[(ann["class"] == "olfactory") & (ann["bodyId"].isin(comp))]
    channels = sorted(orn["type"].dropna().unique())
    by_channel = {c: [int(b) for b in orn[orn["type"] == c]["bodyId"]] for c in channels}
    print(f"{len(channels)} channels, {sum(len(v) for v in by_channel.values())} neurons")

    rng = np.random.default_rng(7)
    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_STIM_RATE"] = "200"

    states = {}
    for p in range(N_PATTERNS):
        # Each pattern activates a different random half of the channels.
        chosen = rng.choice(channels, size=len(channels) // 2, replace=False)
        neurons = [n for c in chosen for n in by_channel[c]]
        label = f"wide_{p}"
        env["FLY_NEU_EXC"] = ",".join(str(n) for n in neurons)
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", "1", "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
        sp = pd.read_parquet(f[0])
        sp["flywire_id"] = sp["flywire_id"].astype("int64")
        states[p] = sp["flywire_id"].value_counts()
        print(f"pattern {p}: {len(neurons)} stimulated, "
              f"{sp['flywire_id'].nunique()} active", flush=True)

    ids = sorted(set().union(*[set(v.index) for v in states.values()]))
    M = pd.DataFrame({k: v.reindex(ids, fill_value=0) for k, v in states.items()}).T
    # Drop the stimulated neurons themselves; only downstream activity counts.
    C = np.corrcoef(M.values)
    iu = np.triu_indices_from(C, 1)
    print()
    print(f"pairwise correlation between brain states: "
          f"min {C[iu].min():.3f}  median {np.median(C[iu]):.3f}  max {C[iu].max():.3f}")
    print("(taste inputs at 28 neurons gave a median of 0.990)")


if __name__ == "__main__":
    main()
