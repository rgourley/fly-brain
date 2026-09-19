"""Turn a stock's indicators into a smell.

Nothing here identifies the company. Two stocks with the same readings
produce the same smell and the fly cannot tell them apart, which is the
point: it learns setups, so a lesson from one name carries to any other
name showing the same shape.

Each feature gets a row of channels covering its range rather than one
channel carrying its value. That way a low reading and a high reading
activate different cells instead of the same cells at different volumes.
Bands overlap, so nearby values smell nearly alike and distant ones share
nothing, which is what makes a learned association generalise.

Channels the data cannot fill stay dark. The fly reads that as nothing
being there, rather than as a reading of zero.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
COMP = ROOT / "data" / "malecns" / "2026_Completeness_malecns.csv"

MAX_HZ = 100.0
WIDTH = 0.75          # band overlap, in units of band spacing


@dataclass(frozen=True)
class Feature:
    """One indicator and the channels that carry it."""

    name: str
    centres: tuple[float, ...]
    label: str

    @property
    def channels(self) -> int:
        return len(self.centres)


def _spread(lo: float, hi: float, n: int) -> tuple[float, ...]:
    return tuple(float(x) for x in np.linspace(lo, hi, n))


# Bands are fixed, not set relative to today's universe. RSI 30 means the
# same thing in March and September, so a lesson learned once stays true.
# The cost is that in a market where everything is oversold, most of the
# board crowds into the same bands.
FEATURES: tuple[Feature, ...] = (
    Feature("rsi", _spread(15, 85, 8), "RSI"),
    Feature("bb_position", _spread(0.0, 1.0, 8), "position in Bollinger band"),
    Feature("distance_from_sma50", _spread(-0.20, 0.20, 8), "distance from 50-day average"),
    Feature("volume_ratio", tuple(float(x) for x in np.geomspace(0.5, 3.0, 8)), "volume vs average"),
    Feature("price_change_5d", _spread(-0.15, 0.15, 8), "5-day return"),
    Feature("sentiment", _spread(-1.0, 1.0, 4), "news sentiment"),
)

# Already categorical in the data, so no banding needed.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "rsi_trend": ("rising", "falling", "flat"),
    "earnings": ("tomorrow", "this week", "later", "none"),
}


def channel_names() -> list[str]:
    """Every channel, in a fixed order that must not change between runs."""
    names = []
    for feature in FEATURES:
        names += [f"{feature.name}[{i}]" for i in range(feature.channels)]
    for field, values in CATEGORIES.items():
        names += [f"{field}={v}" for v in values]
    return names


def smell(reading: dict[str, object]) -> dict[str, float]:
    """Turn one stock's readings into channel activations, 0 to 1.

    A value lights the band it falls in and partly lights its neighbours.
    Missing values leave their channels dark.
    """
    out: dict[str, float] = {}
    for feature in FEATURES:
        value = reading.get(feature.name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        centres = np.asarray(feature.centres, dtype=float)
        spacing = float(np.mean(np.diff(centres))) or 1.0
        strength = np.exp(-(((float(value) - centres) / (WIDTH * spacing)) ** 2))
        peak = strength.max()
        if peak <= 0:
            continue
        for i, s in enumerate(strength / peak):
            if s > 0.02:
                out[f"{feature.name}[{i}]"] = float(s)

    for field, values in CATEGORIES.items():
        got = reading.get(field)
        if got in values:
            out[f"{field}={got}"] = 1.0
    return out


def load_channel_map() -> dict[str, list[int]]:
    """Assign each channel a glomerulus, and list that glomerulus's cells.

    The assignment is by sorted channel name, so it is the same every run.
    """
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    keep = set(pd.read_csv(COMP, index_col=0).index)
    ann = ann[ann["bodyId"].isin(keep)]
    orn = ann[ann["class"] == "olfactory"]
    glomeruli = sorted(orn["type"].dropna().unique())

    names = channel_names()
    if len(names) > len(glomeruli):
        raise ValueError(f"{len(names)} channels needed, {len(glomeruli)} available")
    return {
        name: [int(b) for b in orn[orn["type"] == glom]["bodyId"]]
        for name, glom in zip(names, glomeruli)
    }


def to_rates(activations: dict[str, float],
             channel_map: dict[str, list[int]]) -> dict[int, float]:
    """Turn channel activations into a firing rate per receptor neuron."""
    rates: dict[int, float] = {}
    for name, strength in activations.items():
        for body_id in channel_map.get(name, ()):
            rates[body_id] = strength * MAX_HZ
    return rates


def describe(activations: dict[str, float]) -> str:
    """Say what a stock smells of, for the feed and for checking by eye."""
    by_feature: dict[str, list[str]] = {}
    for name, strength in sorted(activations.items(), key=lambda kv: -kv[1]):
        head = name.split("[")[0].split("=")[0]
        by_feature.setdefault(head, []).append(f"{name} {strength:.2f}")
    labels = {f.name: f.label for f in FEATURES}
    lines = []
    for head, parts in by_feature.items():
        lines.append(f"{labels.get(head, head)}: " + ", ".join(parts[:2]))
    return "\n".join(lines)
