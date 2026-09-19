"""Work out which part of the visual field each readout neuron watches.

Only 15 columnar cell types carry retinotopic coordinates. The motion
detectors and object detectors do not, even though the cells feeding them
do. This walks back along the wiring: a neuron watches wherever its inputs
watch, weighted by how many synapses come from each.

The result is cached, because it only changes if the connectome does.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "malecns" / "body-annotations.feather"
WEIGHTS = ROOT / "data" / "malecns" / "weights.feather"
CACHE = ROOT / "data" / "malecns" / "neuron_regions.parquet"


def build(grid, rebuild: bool = False) -> pd.DataFrame:
    """Return a body_id to region map for every neuron that has one.

    Neurons with coordinates are placed directly. Everything else inherits
    the region of whichever placed neurons feed it most strongly. Neurons
    with no placed input are left out, which is correct: they do not watch
    one part of the visual field.
    """
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)

    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    ann["side"] = ann["instance"].str.extract(r"_(L|R)$")[0]

    placed = ann.dropna(subset=["assignedOlHex1", "assignedOlHex2"]).copy()
    placed["hex1"] = placed["assignedOlHex1"].astype(int)
    placed["hex2"] = placed["assignedOlHex2"].astype(int)
    placed = placed[placed["side"] == "R"].merge(
        grid.columns[["hex1", "hex2", "region"]], on=["hex1", "hex2"], how="inner"
    )[["bodyId", "region"]].rename(columns={"bodyId": "body_id"})

    weights = pd.read_feather(WEIGHTS, columns=["body_pre", "body_post", "weight"])
    edges = weights.merge(placed, left_on="body_pre", right_on="body_id", how="inner")

    # Sum synapses from each region onto each downstream neuron, then take
    # whichever region wins.
    totals = edges.groupby(["body_post", "region"])["weight"].sum().reset_index()
    winner = totals.sort_values("weight").drop_duplicates("body_post", keep="last")

    # How lopsided the win is, so callers can drop neurons that watch
    # everywhere rather than somewhere.
    overall = totals.groupby("body_post")["weight"].sum().rename("total")
    winner = winner.merge(overall, on="body_post")
    winner["share"] = winner["weight"] / winner["total"]

    inherited = winner.rename(columns={"body_post": "body_id"})[
        ["body_id", "region", "share"]
    ]
    placed = placed.assign(share=1.0)
    out = pd.concat([placed, inherited]).drop_duplicates("body_id", keep="first")
    out["region"] = out["region"].astype(int)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return out


def readout_cells(pattern: str, grid, min_share: float = 0.4) -> pd.DataFrame:
    """Return cells of a type matching `pattern`, tagged with their region.

    `min_share` drops cells whose input is spread evenly over the whole
    visual field. Those cells cannot tell you where anything is.
    """
    regions = build(grid)
    ann = pd.read_feather(ANN).drop_duplicates("bodyId")
    ann = ann[ann["type"].fillna("").str.match(pattern)]
    merged = ann.merge(regions, left_on="bodyId", right_on="body_id", how="inner")
    merged = merged[merged["share"] >= min_share]
    return merged[["body_id", "type", "region", "share"]]
