"""Ask which chart the fly's motion system attends to.

Eight regions, eight different price behaviours, animated. Each motion
detector is assigned to the patch of visual field it watches, so the region
with the strongest response is the fly's pick.

A still board runs as a control. If the still board scores the same as the
moving one, the animation is not reaching the motion detectors and nothing
below it means anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_vision import Bar, N_REGIONS, animate_board, board_to_rates, load_grid, price_scale
from fly_regions import readout_cells
from fly_brian import BrianFly

FRAMES = 16
MS_PER_FRAME = 25.0
TRIALS = 4

BEHAVIOURS = {
    0: ("strong up", 0.030, 0.010),
    1: ("strong down", -0.030, 0.010),
    2: ("flat quiet", 0.000, 0.003),
    3: ("flat choppy", 0.000, 0.035),
    4: ("drift up", 0.008, 0.008),
    5: ("drift down", -0.008, 0.008),
    6: ("up volatile", 0.020, 0.030),
    7: ("down volatile", -0.020, 0.030),
}


def synth(n: int, trend: float, vol: float, rng: np.random.Generator) -> list[Bar]:
    price, bars = 100.0, []
    for _ in range(n):
        o = price
        c = o * (1 + trend + rng.normal(0, vol))
        bars.append(Bar(o, max(o, c) * (1 + vol), min(o, c) / (1 + vol), c))
        price = c
    return bars


def score(counts: np.ndarray, cells: pd.DataFrame,
          index_of: dict[int, int]) -> pd.DataFrame:
    rows = []
    for region, group in cells.groupby("region"):
        idx = [index_of[int(b)] for b in group["body_id"] if int(b) in index_of]
        if not idx:
            continue
        total = int(counts[idx].sum())
        rows.append({"region": int(region),
                     "behaviour": BEHAVIOURS[int(region)][0],
                     "cells": len(idx),
                     "spikes": total,
                     "per_cell": round(total / len(idx), 2)})
    return pd.DataFrame(rows).sort_values("per_cell", ascending=False)


def main() -> None:
    grid = load_grid()
    cells = readout_cells(r"T[45][a-d]$", grid)
    print(f"{len(cells)} motion detectors placed in regions")

    fly = BrianFly()
    index_of = fly.index_of

    for label, seed in (("run 1", 101), ("run 2", 202)):
        rng = np.random.default_rng(seed)
        board = {r: synth(grid.width + FRAMES, trend, vol, rng)
                 for r, (_, trend, vol) in BEHAVIOURS.items()}
        frames = animate_board(board, grid, FRAMES)

        totals = np.zeros(fly.n, dtype=np.int64)
        for trial in range(TRIALS):
            totals += fly.show_sequence(frames, MS_PER_FRAME, seed=seed + trial)
        table = score(totals / TRIALS, cells, index_of)
        print(f"\n--- {label} ---")
        print(table.to_string(index=False))
        print(f"winner: {table.iloc[0]['behaviour']}")

        if label == "run 1":
            scales = {r: price_scale(b) for r, b in board.items()}
            still = board_to_rates({r: b[:grid.width] for r, b in board.items()},
                                   grid, scales)
            frozen = [still] * FRAMES
            control = fly.show_sequence(frozen, MS_PER_FRAME, seed=seed)
            ctrl = score(control, cells, index_of)
            moving = table.set_index("region")["per_cell"]
            static = ctrl.set_index("region")["per_cell"]
            print("\nstill-board control (same charts, not moving):")
            print(ctrl.to_string(index=False))
            print(f"\nmoving vs still, correlation across regions: "
                  f"{np.corrcoef(moving.sort_index(), static.sort_index())[0,1]:.3f}")


if __name__ == "__main__":
    main()
