"""Test whether lateralized wind input moves the steering readout.

Real flies get odor direction from wind, sensed as a difference in
displacement between the two antennae. This drives the wind_gravity
mechanosensory population on one side at a time and reads the descending
and motor output on each side.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"


def main() -> None:
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    comp = set(pd.read_csv(ROOT / "data/malecns/2026_Completeness_malecns.csv",
                           index_col=0).index)
    w = ann[(ann["subclass"] == "wind_gravity") & (ann["bodyId"].isin(comp))].copy()
    w["side"] = w["instance"].str.extract(r"_(L|R)$")[0]
    wind = {s: [int(b) for b in w[w["side"] == s]["bodyId"]] for s in "LR"}

    dna = {s: int(ann[ann["instance"] == f"DNa02_{s}"]["bodyId"].iloc[0]) for s in "LR"}
    dn = ann[ann["superclass"] == "descending_neuron"].copy()
    dn["side"] = dn["instance"].str.extract(r"_(L|R)$")[0]
    dn_ids = {s: set(dn[dn["side"] == s]["bodyId"]) for s in "LR"}
    mot = ann[ann["superclass"] == "vnc_motor"].copy()
    mot["side"] = mot["instance"].str.extract(r"_(L|R)$")[0]
    mot_ids = {s: set(mot[mot["side"] == s]["bodyId"]) for s in "LR"}

    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(ROOT / "data/malecns/2026_Completeness_malecns.csv")
    env["FLY_CONN_PATH"] = str(ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
    env["FLY_STIM_RATE"] = "200"

    for stim in "LR":
        label = f"wind_{stim}"
        env["FLY_NEU_EXC"] = ",".join(str(n) for n in wind[stim])
        subprocess.run(
            [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
             "--t_run", "1", "--n_run", "2", "--run-label", label],
            cwd=ROOT, env=env, capture_output=True, check=True,
        )
        f = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
        sp = pd.read_parquet(f[0])
        sp["flywire_id"] = sp["flywire_id"].astype("int64")
        n = sp["trial"].nunique()
        counts = sp["flywire_id"].value_counts() / n

        def pop(ids: set[int]) -> float:
            return float(counts.reindex(list(ids)).fillna(0).sum())

        d_l, d_r = counts.get(dna["L"], 0), counts.get(dna["R"], 0)
        dn_l, dn_r = pop(dn_ids["L"]), pop(dn_ids["R"])
        m_l, m_r = pop(mot_ids["L"]), pop(mot_ids["R"])
        print(f"wind {stim} ({len(wind[stim])} neurons): "
              f"DNa02 L={d_l:.1f} R={d_r:.1f} | "
              f"DN L={dn_l:.0f} R={dn_r:.0f} ({(dn_l-dn_r)/max(dn_l+dn_r,1)*100:+.1f}%) | "
              f"motor L={m_l:.0f} R={m_r:.0f} ({(m_l-m_r)/max(m_l+m_r,1)*100:+.1f}%)",
              flush=True)


if __name__ == "__main__":
    main()
