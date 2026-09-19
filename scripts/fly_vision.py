"""Draw a board of candle charts into the fly's visual field.

The right eye has 892 columns laid out on a hex grid. Each column holds one
L1, one L2 and one L3 cell, which are the first cells the photoreceptors
feed. This module splits that grid into eight regions, draws one ticker's
candles in each, and returns a firing rate for every input cell.

The caller supplies bars. Nothing here touches the market.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
COMP = ROOT / "data" / "malecns" / "2026_Completeness_malecns.csv"

INPUT_TYPES = ("L1", "L2", "L3")
EYE = "R"
REGIONS_ACROSS = 4
REGIONS_DOWN = 2
N_REGIONS = REGIONS_ACROSS * REGIONS_DOWN
MAX_HZ = 200.0
# Every chart is drawn against this shared range, in fractional change.
PCT_RANGE = 0.15


@dataclass(frozen=True)
class Bar:
    """One price bar."""

    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Grid:
    """The fly's visual columns, each tagged with a board region."""

    columns: pd.DataFrame      # hex1, hex2, region, col_x, col_y
    cells: pd.DataFrame        # body_id, hex1, hex2
    width: int                 # columns across one region
    height: int                # rows down one region


def load_grid() -> Grid:
    """Read the retinotopic map for one eye and assign each column a region."""
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    keep = set(pd.read_csv(COMP, index_col=0).index)
    ann = ann[ann["bodyId"].isin(keep)]

    cells = ann[ann["type"].isin(INPUT_TYPES)].dropna(
        subset=["assignedOlHex1", "assignedOlHex2"]
    ).copy()
    cells["side"] = cells["instance"].str.extract(r"_(L|R)$")[0]
    cells = cells[cells["side"] == EYE]
    cells = cells.rename(
        columns={"assignedOlHex1": "hex1", "assignedOlHex2": "hex2",
                 "bodyId": "body_id"}
    )[["body_id", "hex1", "hex2"]]
    cells["hex1"] = cells["hex1"].astype(int)
    cells["hex2"] = cells["hex2"].astype(int)

    columns = cells[["hex1", "hex2"]].drop_duplicates().reset_index(drop=True)

    # The eye is roughly elliptical, so equal rectangles give very unequal
    # column counts and a ticker's position would decide the contest. Split
    # by rank instead, so every region holds the same number of columns.
    columns["band"] = _rank_split(columns["hex2"], REGIONS_DOWN)
    columns["slot"] = -1
    for band in range(REGIONS_DOWN):
        mask = columns["band"] == band
        columns.loc[mask, "slot"] = _rank_split(columns.loc[mask, "hex1"],
                                                REGIONS_ACROSS)
    columns["region"] = columns["band"] * REGIONS_ACROSS + columns["slot"]

    # Position inside each region, measured against that region's own extent.
    width = height = 0
    for region in range(N_REGIONS):
        mask = columns["region"] == region
        x = _scale_to_index(columns.loc[mask, "hex1"])
        y = _scale_to_index(columns.loc[mask, "hex2"])
        columns.loc[mask, "col_x"] = x
        columns.loc[mask, "col_y"] = y
        width = max(width, int(x.max()) + 1)
        height = max(height, int(y.max()) + 1)

    columns["col_x"] = columns["col_x"].astype(int)
    columns["col_y"] = columns["col_y"].astype(int)
    return Grid(columns=columns, cells=cells, width=width, height=height)


def _rank_split(values: pd.Series, parts: int) -> np.ndarray:
    """Cut a series into equally sized groups, ordered low to high."""
    order = values.rank(method="first").to_numpy()
    return np.minimum(((order - 1) / len(values) * parts).astype(int), parts - 1)


def _scale_to_index(values: pd.Series) -> pd.Series:
    """Map values onto 0..n so a region's own extent fills the chart."""
    lo, hi = values.min(), values.max()
    if hi == lo:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return ((values - lo) / (hi - lo) * (hi - lo)).round().astype(int)


def draw_candles(bars: list[Bar], width: int, height: int,
                 scale: tuple[float, float] | None = None) -> np.ndarray:
    """Draw candles as a brightness image, 0 dark to 1 bright.

    One column of the image per bar. The body is bright and the wick is dim,
    so the fly sees a shape rather than a line.

    Pass `scale` as (low, high) to fix the vertical axis. An animation must
    do this. If each frame scales to its own range, the chart slides up and
    down on its own and the fly sees motion the price never made.
    """
    image = np.zeros((height, width), dtype=np.float32)
    if not bars:
        return image

    bars = bars[-width:]
    if scale is None:
        lows = min(b.low for b in bars)
        highs = max(b.high for b in bars)
    else:
        lows, highs = scale
    span = highs - lows
    if span <= 0:
        return image

    def row_of(price: float) -> int:
        # Higher price sits nearer the top of the image. Anything beyond the
        # shared range pins to the edge. Without the clamp a negative index
        # wraps and a strong riser gets drawn along the bottom.
        frac = (price - lows) / span
        row = int(round((1.0 - frac) * (height - 1)))
        return max(0, min(height - 1, row))

    for x, bar in enumerate(bars):
        top, bottom = row_of(bar.high), row_of(bar.low)
        image[top:bottom + 1, x] = 0.35
        body_top, body_bottom = sorted((row_of(bar.open), row_of(bar.close)))
        image[body_top:body_bottom + 1, x] = 1.0
    return image


def to_relative(bars: list[Bar]) -> list[Bar]:
    """Express a series as fractional change from its first open.

    Without this every chart scales to its own high and low, so a stock
    that moved 0.1% and one that moved 40% draw the same picture, and a
    flat series stretches its own tiny wicks across the whole frame.
    """
    if not bars:
        return []
    base = bars[0].open
    if base == 0:
        return bars
    return [
        Bar((b.open - base) / base, (b.high - base) / base,
            (b.low - base) / base, (b.close - base) / base)
        for b in bars
    ]


def price_scale(bars: list[Bar]) -> tuple[float, float]:
    """Return the shared vertical range every chart is drawn against."""
    return (-PCT_RANGE, PCT_RANGE)


def animate_board(board: dict[int, list[Bar]], grid: Grid,
                  frames: int) -> list[dict[int, float]]:
    """Scroll each region's chart and return the firing rates per frame.

    Frame t shows bars t through t + width for every region, so the candles
    walk leftwards as time advances and the price line moves wherever the
    price went. Each region keeps one vertical scale for the whole run.

    Every region needs at least `frames + grid.width` bars.
    """
    sequence = []
    for t in range(frames):
        window = {
            region: bars[t:t + grid.width]
            for region, bars in board.items()
        }
        sequence.append(board_to_rates(window, grid))
    return sequence


def board_to_rates(board: dict[int, list[Bar]], grid: Grid,
                   scales: dict[int, tuple[float, float]] | None = None,
                   ) -> dict[int, float]:
    """Turn a region-to-bars mapping into a firing rate per input cell."""
    scale = (-PCT_RANGE, PCT_RANGE)
    images = {
        region: draw_candles(to_relative(bars), grid.width, grid.height, scale)
        for region, bars in board.items()
    }

    brightness = {}
    for row in grid.columns.itertuples():
        image = images.get(row.region)
        if image is None:
            continue
        y = min(row.col_y, image.shape[0] - 1)
        x = min(row.col_x, image.shape[1] - 1)
        brightness[(row.hex1, row.hex2)] = float(image[y, x])

    rates: dict[int, float] = {}
    for cell in grid.cells.itertuples():
        value = brightness.get((cell.hex1, cell.hex2), 0.0)
        if value > 0:
            rates[int(cell.body_id)] = value * MAX_HZ
    return rates


def render_ascii(board: dict[int, list[Bar]], grid: Grid) -> str:
    """Draw the board as text, to check what the fly is being shown."""
    shades = " .:-=+*#%@"
    out = []
    for region in range(N_REGIONS):
        image = draw_candles(board.get(region, []), grid.width, grid.height)
        rows = [
            "".join(shades[min(int(v * (len(shades) - 1)), len(shades) - 1)]
                    for v in line)
            for line in image
        ]
        out.append(f"region {region}:\n" + "\n".join(rows))
    return "\n\n".join(out)
