"""Test whether the model can tell a rising smell from a fading one.

Real flies navigate by moving and comparing the smell now against the
smell a moment ago. That needs the network to carry history. This drives
one ramp up and one ramp down in the same run and compares the network
state at the moment each passes through the same concentration.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from run_pytorch import MODEL_PARAMS, DT, TorchModel, get_hash_tables, get_weights

COMP = ROOT / "data/malecns/2026_Completeness_malecns.csv"
CONN = ROOT / "data/malecns/2026_Connectivity_malecns.parquet"
PEAK_HZ = 200.0
STEPS = 4000          # 400 ms at dt = 0.1 ms
CROSS = STEPS // 2    # both ramps pass the same rate here


def main() -> None:
    flyid2i, _ = get_hash_tables(COMP)
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    comp_ids = set(flyid2i)
    orn = ann[(ann["class"] == "olfactory") & (ann["bodyId"].isin(comp_ids))]
    idx = torch.tensor([flyid2i[int(b)] for b in orn["bodyId"]], dtype=torch.long)
    print(f"{len(idx)} olfactory neurons driven, {len(flyid2i)} total")

    weights = get_weights(CONN, COMP, ROOT / "data/malecns", csr=False)
    model = TorchModel(batch=2, size=len(flyid2i), dt=DT,
                       params=MODEL_PARAMS, weights=weights, device="cpu")
    state = model.state_init()

    # Row 0 ramps up, row 1 ramps down. They cross at the midpoint.
    up = np.linspace(0.0, PEAK_HZ, STEPS)
    down = np.linspace(PEAK_HZ, 0.0, STEPS)

    spikes = np.zeros((2, STEPS), dtype=np.int64)
    rates = torch.zeros((2, len(flyid2i)), dtype=torch.float32)
    for t in range(STEPS):
        rates.zero_()
        rates[0, idx] = float(up[t])
        rates[1, idx] = float(down[t])
        state = model(rates, *state)
        s = state[2]
        spikes[0, t] = int(s[0].sum())
        spikes[1, t] = int(s[1].sum())

    win = 200  # 20 ms around the crossing
    lo, hi = CROSS - win // 2, CROSS + win // 2
    up_cross = spikes[0, lo:hi].sum()
    down_cross = spikes[1, lo:hi].sum()
    print()
    print(f"same concentration ({PEAK_HZ/2:.0f} Hz), 20 ms window:")
    print(f"  rising  : {up_cross:,} spikes")
    print(f"  falling : {down_cross:,} spikes")
    diff = abs(up_cross - down_cross) / max((up_cross + down_cross) / 2, 1) * 100
    print(f"  difference: {diff:.1f}%")
    print()
    print(f"whole run totals   rising {spikes[0].sum():,}   falling {spikes[1].sum():,}")
    np.save(ROOT / "data/results/ramp_spikes.npy", spikes)


if __name__ == "__main__":
    main()
