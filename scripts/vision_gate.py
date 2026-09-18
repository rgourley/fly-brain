"""Test whether the fly's object-detecting neurons track where something is.

The board design puts one ticker in each region of the visual field. That
only works if lighting up one region produces a different response from
lighting up another. This stimulates the first-order visual cells in two
separate regions and compares the lobula columnar output.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
INPUT_TYPES = ["L1", "L2"]   # direct targets of the photoreceptors


def main() -> None:
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    comp = set(pd.read_csv(ROOT / "data/malecns/2026_Completeness_malecns.csv",
                           index_col=0).index)
    ann = ann[ann["bodyId"].isin(comp)]

    vis = ann[ann["type"].isin(INPUT_TYPES)].dropna(subset=["assignedOlHex1"])
    lo, hi = vis["assignedOlHex1"].min(), vis["assignedOlHex1"].max()
    third = (hi - lo) / 3
    left_patch = vis[vis["assignedOlHex1"] < lo + third]
    right_patch = vis[vis["assignedOlHex1"] > hi - third]
    print(f"{len(vis)} input cells over hex1 {lo:.0f}-{hi:.0f}")
    print(f"patch A: {len(left_patch)} cells   patch B: {len(right_patch)} cells")

    lc = ann[ann["type"].fillna("").str.match(r"LC\d+")]
    lc_ids = set(int(b) for b in lc["bodyId"])
    lc_type = ann.set_index("bodyId")["type"]

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_STIM_RATE"] = "200"

    results = {}
    for name, patch in (("A", left_patch), ("B", right_patch)):
        label = f"vis_{name}"
        env["FLY_NEU_EXC"] = ",".join(str(int(b)) for b in patch["bodyId"])
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", "2", "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
        sp = pd.read_parquet(f[0])
        sp["flywire_id"] = sp["flywire_id"].astype("int64")
        counts = sp["flywire_id"].value_counts() / sp["trial"].nunique()
        results[name] = counts
        total = float(counts.reindex(list(lc_ids)).fillna(0).sum())
        print(f"patch {name}: {sp['flywire_id'].nunique()} neurons active, "
              f"LC total {total:,.0f}", flush=True)

    ids = results["A"].index.union(results["B"].index)
    A = results["A"].reindex(ids, fill_value=0)
    B = results["B"].reindex(ids, fill_value=0)
    lc_mask = ids.isin(lc_ids)
    import numpy as np
    print()
    print(f"whole-brain correlation A vs B: {np.corrcoef(A, B)[0,1]:.3f}")
    if lc_mask.sum():
        print(f"LC-only correlation A vs B:     "
              f"{np.corrcoef(A[lc_mask], B[lc_mask])[0,1]:.3f}")
    d = pd.DataFrame({"A": A[lc_mask], "B": B[lc_mask]})
    d["type"] = lc_type.reindex(d.index).values
    g = d.groupby("type")[["A", "B"]].sum()
    g["diff_pct"] = ((g["A"] - g["B"]) / (g[["A", "B"]].sum(axis=1) / 2) * 100).round(1)
    print()
    print(g.sort_values("diff_pct", key=abs, ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
