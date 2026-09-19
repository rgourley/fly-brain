"""Does the fly learn a smell and generalise to similar ones?

This is the textbook experiment, run on the simulation. Reward one smell
repeatedly and the response to it should climb. A slightly different smell
should climb less, an unrelated one not at all. Punish instead and the
whole gradient should invert.

If that curve does not appear, nothing else measured here means anything,
because the mechanism underneath is not working.
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

LEARNING_RATE = 0.05
PAIRINGS = 8

# One rewarded setup, then the same setup walked steadily away from it, so
# the response can be measured against how different the smell has become.
TRAINED = {"rsi": 28.0, "bb_position": 0.12, "distance_from_sma50": -0.09,
           "volume_ratio": 2.1, "price_change_5d": -0.07, "rsi_trend": "falling"}
AWAY = {"rsi": 78.0, "bb_position": 0.92, "distance_from_sma50": 0.11,
        "volume_ratio": 1.0, "price_change_5d": 0.09, "rsi_trend": "rising"}
STEPS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)


def blend(a: dict, b: dict, t: float) -> dict:
    """A setup t of the way from a to b."""
    out = {}
    for key, value in a.items():
        if isinstance(value, str):
            out[key] = value if t < 0.5 else b[key]
        else:
            out[key] = value + (b[key] - value) * t
    return out


def patterns(fly: BrianFly, kc_index: list[int]) -> dict[str, np.ndarray]:
    channel_map = load_channel_map()
    out = {}
    for t in STEPS:
        setup = blend(TRAINED, AWAY, t)
        counts = fly.show_sequence([to_rates(smell(setup), channel_map)],
                                   ms_per_frame=50.0, seed=77)
        out[f"{t:.2f}"] = counts[kc_index] > 0
    return out


def main() -> None:
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    kc_ids = [int(b) for b in ann[ann["type"].fillna("").str.match(r"KC")]["bodyId"]]
    fly = BrianFly()   # sliced circuit and retuned APL are now the defaults
    index = [fly.index_of[b] for b in kc_ids if b in fly.index_of]
    body_ids = [fly.body_ids[i] for i in index]

    pats = patterns(fly, index)
    trained = pats["0.00"]
    print("distance  cells  shared with trained")
    for name, p in pats.items():
        shared = (p & trained).sum() / max((p | trained).sum(), 1)
        print(f"{name:>8s}  {int(p.sum()):5d}  {shared * 100:3.0f}%")

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

    def run(sign: int) -> list[dict[str, float]]:
        r, p = to_reward.copy(), to_punish.copy()
        history = []
        for _ in range(PAIRINGS + 1):
            history.append({
                name: float(r[pat].sum() - p[pat].sum())
                for name, pat in pats.items()
            })
            cells = pats["0.00"]
            if sign > 0:
                p[cells] *= (1.0 - LEARNING_RATE)
            else:
                r[cells] *= (1.0 - LEARNING_RATE)
        return history

    for label, sign in (("REWARD", 1), ("PUNISH", -1)):
        history = run(sign)
        base = history[0]
        end = history[-1]
        print(f"\n--- {label} the trained smell {PAIRINGS} times ---")
        print("distance   shift in verdict   as % of the trained smell's shift")
        peak = abs(end["0.00"] - base["0.00"]) or 1.0
        for name in pats:
            shift = end[name] - base[name]
            print(f"{name:>8s}   {shift:+12.3f}   {abs(shift) / peak * 100:8.0f}%")


if __name__ == "__main__":
    main()
