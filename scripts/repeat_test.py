"""Check that the fly's pick is stable, and that it is picking the chart.

Two things can look identical in one run. The fly might be reacting to the
shape of a price path, or it might simply favour one patch of its visual
field. So the same eight charts run twice, rotated to different positions
the second time. If the same chart wins both, it is reading the chart. If
the same position wins both, it is reading the wiring.

Every arrangement runs several times, because the feeding circuit turned
out to flip on identical input and nothing here gets believed on one run.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_vision import Bar, N_REGIONS, board_to_rates, load_grid, price_scale
from fly_regions import readout_cells
from fly_brian import BrianFly

TRIALS = 6
MS = 200.0
ROTATION = 3          # how far the charts shift on the second arrangement


def path(values: list[float]) -> list[Bar]:
    bars = []
    for a, b in zip(values, values[1:]):
        bars.append(Bar(a, max(a, b) * 1.005, min(a, b) * 0.995, b))
    return bars


def shapes(n: int) -> dict[str, list[Bar]]:
    half = n // 2
    return {
        "steady rise": path(list(np.linspace(100, 112, n))),
        "steady fall": path(list(np.linspace(100, 88, n))),
        "V bottom": path(list(np.linspace(100, 90, half)) + list(np.linspace(90, 104, n - half))),
        "peak": path(list(np.linspace(100, 110, half)) + list(np.linspace(110, 96, n - half))),
        "choppy": path([100 + 4 * (i % 2) for i in range(n)]),
        "flat": path([100.0] * n),
        "step up": path([100.0] * half + [108.0] * (n - half)),
        "late spike": path([100.0] * (n - 3) + [100.0, 105.0, 111.0]),
    }


def run_board(fly: BrianFly, grid, cells: pd.DataFrame,
              layout: dict[int, str], library: dict[str, list[Bar]],
              seed: int) -> pd.Series:
    board = {region: library[name] for region, name in layout.items()}
    scales = {r: price_scale(b) for r, b in board.items()}
    rates = board_to_rates(board, grid, scales)
    counts = fly.show_sequence([rates], ms_per_frame=MS, seed=seed)

    per_region = {}
    for region, group in cells.groupby("region"):
        idx = [fly.index_of[int(b)] for b in group["body_id"] if int(b) in fly.index_of]
        per_region[int(region)] = counts[idx].sum() / len(idx)
    return pd.Series(per_region).sort_index()


def main() -> None:
    grid = load_grid()
    cells = readout_cells(r"(Mi|Tm|L)\d+", grid, min_share=0.5)
    print(f"{len(cells)} readout cells; per region "
          f"{cells.groupby('region').size().to_dict()}")

    library = shapes(grid.width + 1)
    names = list(library)
    fly = BrianFly()

    arrangements = {
        "A": {r: names[r] for r in range(N_REGIONS)},
        "B": {r: names[(r + ROTATION) % N_REGIONS] for r in range(N_REGIONS)},
    }

    picks: dict[str, Counter] = {}
    for label, layout in arrangements.items():
        scores = []
        for trial in range(TRIALS):
            scores.append(run_board(fly, grid, cells, layout, library,
                                    seed=1000 + trial))
            print(f"  {label} trial {trial + 1}/{TRIALS} done", flush=True)
        frame = pd.DataFrame(scores)
        winners = Counter(layout[int(r)] for r in frame.idxmax(axis=1))
        picks[label] = winners

        table = pd.DataFrame({
            "chart": [layout[r] for r in frame.columns],
            "mean": frame.mean().round(1).values,
            "sd": frame.std().round(1).values,
        }, index=frame.columns)
        table.index.name = "region"
        print(f"\n--- arrangement {label} ---")
        print(table.sort_values("mean", ascending=False).to_string())
        print(f"winners across {TRIALS} trials: {dict(winners)}")

    print("\n=== does the chart travel with its position? ===")
    for label, counter in picks.items():
        top, count = counter.most_common(1)[0]
        print(f"{label}: {top} won {count}/{TRIALS}")
    a_top = picks["A"].most_common(1)[0][0]
    b_top = picks["B"].most_common(1)[0][0]
    print(f"\nsame chart wins in both arrangements: {a_top == b_top}"
          f"  ({a_top!r} vs {b_top!r})")


if __name__ == "__main__":
    main()
