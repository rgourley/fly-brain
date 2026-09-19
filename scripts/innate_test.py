"""What does the fly prefer before it has learned anything?

Untrained, every synapse still holds the value the connectome gave it, and
those values are not symmetric. Some Kenyon cells feed the reward-side
output neurons more strongly, others the punishment side. So an untrained
fly still produces a verdict for each smell, set by the wiring rather than
by experience. Real flies are the same: born finding some odours attractive
and others aversive.

This runs a spread of setups through an untrained fly and asks whether the
verdicts differ at all. If they are flat, day one really is a coin flip. If
they spread, the fly has preferences it was born with, and they can be
published before it has traded anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_smell import load_channel_map, smell, to_rates
from fly_brian import BrianFly
from dopamine_test import output_groups

SETUPS = {
    "oversold, heavy volume, near lows":
        dict(rsi=24, bb_position=0.08, distance_from_sma50=-0.12,
             volume_ratio=2.4, price_change_5d=-0.09, rsi_trend="falling"),
    "overbought breakout":
        dict(rsi=76, bb_position=0.93, distance_from_sma50=0.13,
             volume_ratio=2.1, price_change_5d=0.11, rsi_trend="rising"),
    "quiet drift up":
        dict(rsi=58, bb_position=0.62, distance_from_sma50=0.03,
             volume_ratio=0.9, price_change_5d=0.02, rsi_trend="rising"),
    "quiet drift down":
        dict(rsi=44, bb_position=0.38, distance_from_sma50=-0.03,
             volume_ratio=0.9, price_change_5d=-0.02, rsi_trend="falling"),
    "dead flat":
        dict(rsi=50, bb_position=0.50, distance_from_sma50=0.00,
             volume_ratio=1.0, price_change_5d=0.00, rsi_trend="flat"),
    "capitulation":
        dict(rsi=14, bb_position=0.02, distance_from_sma50=-0.19,
             volume_ratio=3.6, price_change_5d=-0.16, rsi_trend="falling"),
    "blow-off top":
        dict(rsi=88, bb_position=0.99, distance_from_sma50=0.18,
             volume_ratio=3.2, price_change_5d=0.15, rsi_trend="rising"),
    "squeeze, no volume":
        dict(rsi=52, bb_position=0.48, distance_from_sma50=0.01,
             volume_ratio=0.4, price_change_5d=0.00, rsi_trend="flat"),
}


def main() -> None:
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    kc_ids = [int(b) for b in ann[ann["type"].fillna("").str.match(r"KC")]["bodyId"]]
    fly = BrianFly()
    index = [fly.index_of[b] for b in kc_ids if b in fly.index_of]
    body_ids = [fly.body_ids[i] for i in index]

    reward, punish, kc_to_mbon, _ = output_groups()
    position = {int(b): i for i, b in enumerate(body_ids)}
    reward_set, punish_set = set(reward), set(punish)
    to_reward = np.zeros(len(body_ids))
    to_punish = np.zeros(len(body_ids))
    for row in kc_to_mbon.itertuples():
        i = position.get(int(row.body_pre))
        if i is None:
            continue
        if int(row.body_post) in reward_set:
            to_reward[i] += row.weight
        elif int(row.body_post) in punish_set:
            to_punish[i] += row.weight
    scale = max(to_reward.max(), to_punish.max(), 1.0)
    to_reward /= scale
    to_punish /= scale

    channel_map = load_channel_map()
    rows = []
    for name, setup in SETUPS.items():
        # Several trials, because the input is Poisson and one run is a sample.
        verdicts = []
        for trial in range(4):
            counts = fly.show_sequence([to_rates(smell(setup), channel_map)],
                                       ms_per_frame=100.0, seed=300 + trial)
            cells = counts[index] > 0
            verdicts.append(float(to_reward[cells].sum() - to_punish[cells].sum()))
        rows.append({"setup": name,
                     "verdict": round(float(np.mean(verdicts)), 3),
                     "sd": round(float(np.std(verdicts)), 3),
                     "cells": int(cells.sum())})
        print(f"  {name}", flush=True)

    table = pd.DataFrame(rows).sort_values("verdict", ascending=False)
    print()
    print(table.to_string(index=False))
    spread = table["verdict"].max() - table["verdict"].min()
    noise = table["sd"].mean()
    print(f"\nspread across setups: {spread:.3f}")
    print(f"average run-to-run noise: {noise:.3f}")
    print(f"spread is {spread / max(noise, 1e-9):.1f}x the noise")


if __name__ == "__main__":
    main()
