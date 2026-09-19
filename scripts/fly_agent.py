"""One trading session for the fly.

The loop, in order:

  settle    ask which positions closed since last time, and credit each
            outcome to the Kenyon cells that were firing when the fly
            chose it, however long ago that was
  forget    drift every synapse a little back toward the connectome, so
            lessons that stop being confirmed fade
  smell     turn each stock on the board into a smell and run it
  decide    read the verdict, buy the best one it does not already hold,
            size off how far ahead it is
  remember  write the synapses and the open positions to disk

Market data and orders go through ClawStreet. Everything else is local.
"""

import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_smell import load_channel_map, smell, to_rates, describe
from fly_brian import BrianFly
from fly_memory import FlyMemory, OpenPosition, from_connectome

PRESENTATION_MS = 50.0
MAX_POSITIONS = 6
BOARD_SIZE = 8
# Spikes are not dollars. This is ours and it is declared.
DOLLARS_PER_VERDICT = 250.0
MAX_POSITION = 2000.0


@dataclass(frozen=True)
class Candidate:
    """One stock on the board, as the fly sees it."""

    symbol: str
    reading: dict
    activations: dict[str, float]
    cells: np.ndarray
    verdict: float


def kenyon_cells(fly: BrianFly) -> tuple[list[int], list[int]]:
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    ids = [int(b) for b in ann[ann["type"].fillna("").str.match(r"KC")]["bodyId"]]
    index = [fly.index_of[b] for b in ids if b in fly.index_of]
    return index, [fly.body_ids[i] for i in index]


def reading_from_history(entry: dict) -> dict:
    """Pull the six features out of one symbol's history response."""
    derived = entry.get("derived") or {}
    rsi = entry.get("rsi") or []
    return {
        "rsi": rsi[-1] if rsi else None,
        "bb_position": derived.get("bb_position"),
        "distance_from_sma50": derived.get("distance_from_sma50"),
        "volume_ratio": derived.get("volume_ratio"),
        "price_change_5d": derived.get("price_change_5d"),
        "rsi_trend": derived.get("rsi_trend"),
    }


def sniff(fly: BrianFly, memory: FlyMemory, kc_index: list[int],
          channel_map: dict, symbol: str, reading: dict,
          seed: int) -> Candidate:
    """Show the fly one stock and record what it thought."""
    activations = smell(reading)
    counts = fly.show_sequence([to_rates(activations, channel_map)],
                               ms_per_frame=PRESENTATION_MS, seed=seed)
    cells = counts[kc_index] > 0
    return Candidate(symbol=symbol, reading=reading, activations=activations,
                     cells=cells, verdict=memory.verdict(cells))


def size(verdict: float, runner_up: float) -> float:
    """How much to buy, from how far ahead the winner is.

    A clear favourite gets a real position. One that barely edges out second
    place gets a token. Untrained, every verdict sits near the same value so
    the margins are small and the fly bets little, which is the right
    behaviour for an animal that knows nothing yet.
    """
    margin = max(verdict - runner_up, 0.0)
    return float(min(margin * DOLLARS_PER_VERDICT, MAX_POSITION))


def run_session(board: dict[str, dict], closed: dict[str, bool],
                dry_run: bool = True) -> dict:
    """One decision. `closed` maps a symbol to whether its trade made money."""
    fly = BrianFly()
    kc_index, kc_bodies = kenyon_cells(fly)
    memory = from_connectome(kc_bodies)
    memory.load()
    started_at = memory.drift()

    settled = []
    for symbol, profitable in closed.items():
        position = memory.close(symbol, profitable)
        if position:
            settled.append((symbol, profitable))

    memory.forget()

    channel_map = load_channel_map()
    seed = int(date.today().strftime("%Y%m%d"))
    candidates = [
        sniff(fly, memory, kc_index, channel_map, symbol,
              reading_from_history(entry), seed + i)
        for i, (symbol, entry) in enumerate(board.items())
    ]
    candidates.sort(key=lambda c: c.verdict, reverse=True)

    held = memory.held()
    pick = next((c for c in candidates if c.symbol not in held), None)
    runner_up = next((c.verdict for c in candidates
                      if pick and c.symbol != pick.symbol), 0.0)

    order = None
    if pick and len(held) < MAX_POSITIONS:
        dollars = size(pick.verdict, runner_up)
        if dollars > 0:
            order = {"symbol": pick.symbol, "dollars": round(dollars, 2),
                     "verdict": round(pick.verdict, 3),
                     "margin": round(pick.verdict - runner_up, 3)}
            if not dry_run:
                memory.open(OpenPosition(
                    symbol=pick.symbol, opened=date.today().isoformat(),
                    qty=0.0, price=0.0,
                    cells=[int(i) for i in np.flatnonzero(pick.cells)],
                    verdict=pick.verdict,
                    smell={k: round(v, 2) for k, v in pick.activations.items()},
                ))

    if not dry_run:
        memory.sessions += 1
        memory.save()

    return {
        "session": memory.sessions,
        "settled": settled,
        "ranking": [(c.symbol, round(c.verdict, 3)) for c in candidates],
        "order": order,
        "held": sorted(memory.held()),
        "drift_before": round(started_at, 5),
        "drift_after": round(memory.drift(), 5),
        "smell_of_pick": describe(pick.activations) if pick else "",
    }
