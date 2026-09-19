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
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_smell import load_channel_map, smell, to_rates, describe
from fly_brian import BrianFly
from fly_memory import FlyMemory, OpenPosition, from_connectome, FLIES

PRESENTATION_MS = 50.0
MAX_POSITIONS = 6
BOARD_SIZE = 8
# Spikes are not dollars. These are ours and they are declared.
# ClawStreet agents start with $100,000; the live run passes real equity in. A position is a share of the
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
    """The session as a list of events, in the order they happened.

    Replays live under data/flies/<id>/replays/, one file per session, named
    by the minute the session ran. index.json lists them, oldest first, so
    the page can find the latest without a directory listing.
    """

    def __init__(self, fly: str, when: datetime, dry: bool = False) -> None:
        self.fly = fly
        self.when = when
        self.dry = dry          # a rehearsal: recorded for the page, nothing traded, nothing learned
        self.events: list[dict] = []

    def add(self, kind: str, **fields: object) -> None:
        self.events.append({"step": len(self.events), "type": kind, **fields})

    def write(self) -> Path:
        folder = FLIES / self.fly / "replays"
        folder.mkdir(parents=True, exist_ok=True)
        name = self.when.strftime("%Y-%m-%dT%H%M") + (".dry" if self.dry else "") + ".json"
        (folder / name).write_text(json.dumps({
            "fly": self.fly, "when": self.when.isoformat(timespec="seconds"),
            "dry": self.dry, "events": self.events}, indent=1))
        index = folder / "index.json"
        listed = json.loads(index.read_text()) if index.exists() else []
        if not self.dry:
            # A real session retires the rehearsals that came before it.
            for old in [n for n in listed if n.endswith(".dry.json")]:
                (folder / old).unlink(missing_ok=True)
            listed = [n for n in listed if not n.endswith(".dry.json")]
        if name not in listed:
            listed.append(name)
        index.write_text(json.dumps(listed))
        return folder / name


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


def bars_from_history(entry: dict, n: int = 20) -> list[list[float]]:
    """The last n candles as [open, high, low, close], for the page's cards."""
    o, h, l, c = (entry.get(k) or [] for k in ("open", "high", "low", "prices"))
    rows = list(zip(o, h, l, c))[-n:]
    return [[float(a), float(b), float(d), float(e)] for a, b, d, e in rows]


def run_session(board: dict[str, dict], closed: dict[str, bool],
                dry_run: bool = True, day: date | None = None,
                fly_id: str = "001", when: datetime | None = None,
                account: float = ACCOUNT, record: bool = False) -> dict:
    """One decision. `closed` maps a symbol to whether its trade made money."""
    when = when or datetime.now(timezone.utc)
    day = day or when.date()
    rec = Recorder(fly_id, when, dry=dry_run)
    fly = BrianFly()
    kc_index, kc_bodies = kenyon_cells(fly)
    memory = from_connectome(kc_bodies, fly_id)
    memory.load()
    rec.add("start", session=memory.sessions + 1, held=sorted(memory.held()),
            drift=round(memory.drift(), 5))
    # Everything the page needs to draw the table comes with the replay.
    rec.add("board", stocks={sym: {"reading": reading_from_history(e),
                                   "price": e.get("current_price"),
                                   "bars": bars_from_history(e)}
                             for sym, e in board.items()})

    settled_cells: dict[str, int] = {}
    for symbol, profitable in closed.items():
        position = memory.close(symbol, profitable)
        if position:
            settled_cells[symbol] = len(position.cells)
            rec.add("settle", symbol=symbol, profitable=profitable,
                    cells=len(position.cells), opened=position.opened,
                    compartment="reward" if profitable else "punishment")

    memory.forget()
    rec.add("forget", drift=round(memory.drift(), 5))

    channel_map = load_channel_map()
    # Brian2 takes a 32-bit seed. The minute of the session, folded to fit.
    seed = int(when.strftime("%y%m%d%H%M")) % (2**32 - 1000)
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
        dollars = size(pick.verdict, runner_up, account)
        if dollars > 0:
            order = {"symbol": pick.symbol, "dollars": round(dollars, 2),
                     "verdict": round(pick.verdict, 3),
                     "margin": round(pick.verdict - runner_up, 3)}
            rec.add("buy", **order, smell=describe(pick.activations))
            if not dry_run:
                memory.open(OpenPosition(
                    symbol=pick.symbol, opened=day.isoformat(),
                    qty=0.0, price=float(board[pick.symbol].get("current_price") or 0.0),
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
    if not dry_run or record:
        replay = rec.write()

    return {
        "session": memory.sessions,
        "lessons": memory.lessons,
        "fly": fly_id,
        "when": when.isoformat(timespec="minutes"),
        "settled": [(s, p) for s, p in closed.items()],
        "ranking": [(c.symbol, round(c.verdict, 3)) for c in candidates],
        "sold": sold,
        "order": order,
        "held": sorted(memory.held()),
        "pending": sorted(p.symbol for p in memory.pending),
        "drift": round(memory.drift(), 5),
        "replay": str(replay) if replay else None,
        "settled_cells": {s: n for s, n in settled_cells.items()},
        "pick_cells": int(pick.cells.sum()) if pick else 0,
        "channels": int(sum(1 for v in pick.activations.values() if v >= 0.5)) if pick else 0,
        "runner_up": runner_up if pick else None,
        "positions": [p.__dict__ for p in memory.positions],
    }
