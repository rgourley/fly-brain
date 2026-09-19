"""One trading session for the fly, recorded so it can be replayed.

The loop, in order:

  settle     credit each closed trade to the Kenyon cells that were firing
             when the fly chose it, however long ago that was
  forget     drift every synapse a little back toward the connectome
  smell      run every stock on the board through the olfactory circuit
  reconsider re-smell what it holds; two sessions running of liking a
             holding less than at purchase, and it sells
  buy        the best thing it does not hold, sized off the margin
  remember   write the synapses, the positions and the replay to disk

Every step writes an event to the replay, so the page can play the session
back exactly as it happened. Nothing is drawn later that did not occur.

Market data and orders go through ClawStreet. Everything else is local.
"""

import json
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
REPLAYS = ROOT / "data" / "replays"
# Spikes are not dollars. These are ours and they are declared.
# ClawStreet agents start with $100,000. A position is a share of the
# account set by how convinced the fly is: its verdict against the strongest
# innate verdict we measured (6.26 for an overbought breakout, untrained).
# A tie with second place halves it, because the fly could not separate
# them. The real run reads cash from the API instead of this constant.
ACCOUNT = 100_000.0
MAX_FRACTION = 0.15
FULL_VERDICT = 6.0


@dataclass(frozen=True)
class Candidate:
    """One stock on the board, as the fly saw it."""

    symbol: str
    reading: dict
    activations: dict[str, float]
    cells: np.ndarray
    verdict: float


class Recorder:
    """The session as a list of events, in the order they happened."""

    def __init__(self, day: date) -> None:
        self.day = day
        self.events: list[dict] = []

    def add(self, kind: str, **fields: object) -> None:
        self.events.append({"step": len(self.events), "type": kind, **fields})

    def write(self) -> Path:
        REPLAYS.mkdir(parents=True, exist_ok=True)
        path = REPLAYS / f"{self.day.isoformat()}.json"
        path.write_text(json.dumps({"date": self.day.isoformat(),
                                    "events": self.events}, indent=1))
        return path


def compose_board(held: set[str], universe: list[str], seed: int,
                  size: int = BOARD_SIZE) -> list[str]:
    """Holdings keep their slots. The rest come fresh from the universe.

    Holdings must be on the board so they get re-smelled and can be sold.
    The cap on positions is below the board size, so at least two new names
    arrive every session and the fly can never get stuck in a closed world.
    """
    rng = np.random.default_rng(seed)
    fresh = [s for s in universe if s not in held]
    picks = list(rng.choice(fresh, size=max(0, size - len(held)), replace=False))
    return sorted(held) + [str(s) for s in picks]


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


def size(verdict: float, runner_up: float, account: float = ACCOUNT) -> float:
    """How much to buy, from how convinced the fly is.

    Conviction is the verdict against the strongest innate verdict measured,
    capped at a share of the account. A tie with second place, closer than
    the measured noise, halves it: the fly liked both and could not choose.
    """
    conviction = min(1.0, max(verdict, 0.0) / FULL_VERDICT)
    dollars = account * MAX_FRACTION * conviction
    if verdict - runner_up < 0.36:
        dollars /= 2
    return float(round(dollars, 2))


def run_session(board: dict[str, dict], closed: dict[str, bool],
                dry_run: bool = True, day: date | None = None) -> dict:
    """One decision. `closed` maps a symbol to whether its trade made money."""
    day = day or date.today()
    rec = Recorder(day)
    fly = BrianFly()
    kc_index, kc_bodies = kenyon_cells(fly)
    memory = from_connectome(kc_bodies)
    memory.load()
    rec.add("start", session=memory.sessions + 1, held=sorted(memory.held()),
            drift=round(memory.drift(), 5))

    for symbol, profitable in closed.items():
        position = memory.close(symbol, profitable)
        if position:
            rec.add("settle", symbol=symbol, profitable=profitable,
                    cells=len(position.cells), opened=position.opened,
                    compartment="reward" if profitable else "punishment")

    memory.forget()
    rec.add("forget", drift=round(memory.drift(), 5))

    channel_map = load_channel_map()
    seed = int(day.strftime("%Y%m%d"))
    held = memory.held()
    candidates: list[Candidate] = []
    for i, (symbol, entry) in enumerate(board.items()):
        c = sniff(fly, memory, kc_index, channel_map, symbol,
                  reading_from_history(entry), seed + i)
        candidates.append(c)
        rec.add("sniff", symbol=symbol, held=symbol in held,
                channels={k: round(v, 3) for k, v in c.activations.items()},
                cells=[int(x) for x in np.flatnonzero(c.cells)],
                verdict=round(c.verdict, 3))

    sold = []
    for c in candidates:
        if c.symbol not in held:
            continue
        then = next(p.verdict for p in memory.positions if p.symbol == c.symbol)
        should_sell = memory.reconsider(c.symbol, c.verdict)
        cooling = next(p.cooling for p in memory.positions if p.symbol == c.symbol)
        rec.add("reconsider", symbol=c.symbol, verdict_then=round(then, 3),
                verdict_now=round(c.verdict, 3), cooling=cooling)
        if should_sell:
            memory.sell(c.symbol)
            sold.append(c.symbol)
            rec.add("sell", symbol=c.symbol,
                    reason="liked it less than at purchase, two sessions running")

    candidates.sort(key=lambda c: c.verdict, reverse=True)
    rec.add("rank", order=[(c.symbol, round(c.verdict, 3)) for c in candidates])

    # Margin is measured against the next stock the fly could actually buy.
    # A holding or a just-sold name can outscore the pick, and it must not
    # zero the position: that happened when S0 was sold and nothing was bought
    # for two sessions.
    taken = memory.held() | {p.symbol for p in memory.pending}
    eligible = [c for c in candidates if c.symbol not in taken]
    pick = eligible[0] if eligible else None
    runner_up = eligible[1].verdict if len(eligible) > 1 else 0.0

    order = None
    if pick and len(memory.held()) < MAX_POSITIONS:
        dollars = size(pick.verdict, runner_up)
        if dollars > 0:
            order = {"symbol": pick.symbol, "dollars": round(dollars, 2),
                     "verdict": round(pick.verdict, 3),
                     "margin": round(pick.verdict - runner_up, 3)}
            rec.add("buy", **order, smell=describe(pick.activations))
            if not dry_run:
                memory.open(OpenPosition(
                    symbol=pick.symbol, opened=day.isoformat(),
                    qty=0.0, price=0.0,
                    cells=[int(i) for i in np.flatnonzero(pick.cells)],
                    verdict=pick.verdict,
                    smell={k: round(v, 2) for k, v in pick.activations.items()},
                ))

    rec.add("end", held=sorted(memory.held()),
            pending=sorted(p.symbol for p in memory.pending),
            drift=round(memory.drift(), 5))

    replay = None
    if not dry_run:
        memory.sessions += 1
        memory.save()
        replay = rec.write()

    return {
        "session": memory.sessions,
        "settled": [(s, p) for s, p in closed.items()],
        "ranking": [(c.symbol, round(c.verdict, 3)) for c in candidates],
        "sold": sold,
        "order": order,
        "held": sorted(memory.held()),
        "pending": sorted(p.symbol for p in memory.pending),
        "drift": round(memory.drift(), 5),
        "replay": str(replay) if replay else None,
    }
