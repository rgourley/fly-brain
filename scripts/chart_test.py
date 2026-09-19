"""Ask whether the fly reacts differently to a rising chart than a falling one.

Same board, same eight regions, same everything except region 0 holds an
uptrend in one run and a downtrend in the other. If the object-detecting
neurons respond the same to both, the fly cannot read a candle chart.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_vision import Bar, N_REGIONS, board_to_rates, load_grid
from fly_run import Fly


def synth(width: int, trend: float, rng: np.random.Generator) -> list[Bar]:
    price, bars = 100.0, []
    for _ in range(width):
        o = price
        c = o * (1 + trend + rng.normal(0, 0.015))
        bars.append(Bar(o, max(o, c) * 1.01, min(o, c) * 0.99, c))
        price = c
    return bars


def main() -> None:
    grid = load_grid()
    rng = np.random.default_rng(11)
    flat = {r: synth(grid.width, 0.0, rng) for r in range(N_REGIONS)}

    up = dict(flat)
    up[0] = synth(grid.width, 0.04, rng)
    down = dict(flat)
    down[0] = synth(grid.width, -0.04, rng)

    fly = Fly(trials=8)
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    lc = ann[ann["type"].fillna("").str.match(r"LC\d+")]
    lc_idx = [fly.flyid2i[int(b)] for b in lc["bodyId"] if int(b) in fly.flyid2i]
    lc_types = ann.set_index("bodyId")["type"]
    print(f"{len(lc_idx)} object-detecting neurons read")

    results = {}
    for name, board in (("rising", up), ("falling", down)):
        rates = board_to_rates(board, grid)
        counts = fly.show(rates, ms=200.0, seed=42)
        results[name] = counts
        per_trial = counts[:, lc_idx].sum(axis=1)
        print(f"{name:8s}: {len(rates)} cells driven, "
              f"LC spikes per trial mean {per_trial.mean():.0f} "
              f"sd {per_trial.std():.0f}", flush=True)

    a = results["rising"][:, lc_idx].mean(axis=0)
    b = results["falling"][:, lc_idx].mean(axis=0)
    print()
    print(f"LC correlation rising vs falling: {np.corrcoef(a, b)[0, 1]:.3f}")

    frame = pd.DataFrame({"rising": a, "falling": b},
                         index=[int(x) for x in lc["bodyId"] if int(x) in fly.flyid2i])
    frame["type"] = lc_types.reindex(frame.index).values
    grouped = frame.groupby("type")[["rising", "falling"]].sum()
    total = grouped.sum(axis=1) / 2
    grouped["diff_pct"] = ((grouped["rising"] - grouped["falling"]) /
                           total.replace(0, np.nan) * 100).round(1)
    print()
    print(grouped[grouped[["rising", "falling"]].sum(axis=1) > 5]
          .sort_values("diff_pct", key=abs, ascending=False).head(10).to_string())


if __name__ == "__main__":
    main()
