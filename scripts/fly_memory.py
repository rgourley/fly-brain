"""What the fly remembers between sessions.

Two numbers per Kenyon cell: how strongly it drives the reward-side output
neurons and the punishment-side ones. Both start at the values the
connectome gives and move only when dopamine arrives.

Also holds the open positions, each with the Kenyon cells that were firing
when it was opened. That is what makes credit assignment honest. When a
position closes weeks later, the profit goes to the cells that were active
at the moment the fly chose it, not to whatever happens to be firing now.

Nothing here is fitted to market outcomes in the machine learning sense.
The synapses move by a fixed rule in response to a single trade.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FLIES = ROOT / "data" / "flies"

# Chosen parameters. Declared here because they are not from the connectome.
LEARNING_RATE = 0.05     # how far one outcome moves the synapses it touches
DECAY_RATE = 0.02        # how far every synapse drifts home each session
# Measured, not chosen: a verdict wanders this much across five sessions on
# a stock whose readings did not move (exit_test.py). A holding is only
# "liked less" when today's verdict is below the purchase verdict by more
# than this. Without it, two noisy sniffs in a row sold SOFI for no reason.
NOISE_FLOOR = 0.36


@dataclass
class OpenPosition:
    """A trade the fly has taken and not yet been told the result of."""

    symbol: str
    opened: str
    qty: float
    price: float
    cells: list[int]      # Kenyon cells firing when the fly chose it
    verdict: float        # what the fly thought of it at purchase
    smell: dict[str, float] = field(default_factory=dict)
    cooling: int = 0      # sessions in a row it has liked this less than at purchase


class FlyMemory:
    """The fly's learned state, loaded at the start of a session and saved at the end."""

    def __init__(self, baseline_reward: np.ndarray, baseline_punish: np.ndarray,
                 body_ids: list[int], home: Path) -> None:
        # Each fly keeps its own synapses and positions under data/flies/<id>/.
        self.home = home
        self.state_path = home / "memory.npz"
        self.positions_path = home / "positions.json"
        self.baseline_reward = baseline_reward
        self.baseline_punish = baseline_punish
        self.body_ids = body_ids
        self.to_reward = baseline_reward.copy()
        self.to_punish = baseline_punish.copy()
        self.sessions = 0
        self.positions: list[OpenPosition] = []
        # Sold but not yet told the result. Kept so the outcome can still be
        # credited to the cells that chose the trade when the fill comes back.
        self.pending: list[OpenPosition] = []

    # ---- persistence -------------------------------------------------

    def load(self) -> None:
        """Restore what the fly learned. A fly that forgets nightly never learns."""
        if self.state_path.exists():
            saved = np.load(self.state_path)
            if len(saved["to_reward"]) == len(self.to_reward):
                self.to_reward = saved["to_reward"]
                self.to_punish = saved["to_punish"]
                self.sessions = int(saved["sessions"])
        if self.positions_path.exists():
            raw = json.loads(self.positions_path.read_text())
            if isinstance(raw, dict):
                self.positions = [OpenPosition(**p) for p in raw.get("open", [])]
                self.pending = [OpenPosition(**p) for p in raw.get("pending", [])]
            else:
                self.positions = [OpenPosition(**p) for p in raw]

    def save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.state_path, to_reward=self.to_reward,
                            to_punish=self.to_punish, sessions=self.sessions)
        self.positions_path.write_text(json.dumps({
            "open": [p.__dict__ for p in self.positions],
            "pending": [p.__dict__ for p in self.pending],
        }, indent=1))

    # ---- the fly's opinion -------------------------------------------

    def verdict(self, cells: np.ndarray) -> float:
        """Reward-side drive minus punishment-side drive, for the cells firing."""
        return float(self.to_reward[cells].sum() - self.to_punish[cells].sum())

    # ---- learning ----------------------------------------------------

    def learn(self, cells: list[int], profitable: bool) -> None:
        """Dopamine depresses the opposite compartment for the cells that fired.

        A win depresses their drive onto the punishment side, so avoidance
        falls and the verdict for that smell rises. A loss does the reverse.
        Only cells that were firing are touched, which is what ties the
        lesson to that particular setup rather than to everything.
        """
        index = np.asarray(cells, dtype=int)
        if index.size == 0:
            return
        if profitable:
            self.to_punish[index] *= (1.0 - LEARNING_RATE)
        else:
            self.to_reward[index] *= (1.0 - LEARNING_RATE)

    def forget(self) -> None:
        """Drift every synapse back toward the value the connectome gave it.

        Without this the rule only ever depresses, so the synapses walk to
        zero and the fly goes numb. Forgetting is also what keeps it adapting:
        a lesson that stops being confirmed fades, while one that keeps paying
        off gets topped up faster than it decays.
        """
        self.to_reward += DECAY_RATE * (self.baseline_reward - self.to_reward)
        self.to_punish += DECAY_RATE * (self.baseline_punish - self.to_punish)

    # ---- positions ---------------------------------------------------

    def open(self, position: OpenPosition) -> None:
        self.positions.append(position)

    def close(self, symbol: str, profitable: bool) -> OpenPosition | None:
        """Settle a trade and credit the outcome to the cells that chose it.

        Looks in both open and pending, because a position the fly decided to
        sell leaves the open list before its fill and result come back.
        """
        for bucket in (self.pending, self.positions):
            for i, p in enumerate(bucket):
                if p.symbol == symbol:
                    self.learn(p.cells, profitable)
                    return bucket.pop(i)
        return None

    def reconsider(self, symbol: str, verdict_today: float) -> bool:
        """Compare a holding against the opinion it was bought on.

        Returns True when the fly has liked it less than at purchase for two
        sessions running. Two, not one, because run-to-run noise is about
        0.36 and a real change in the setup is about 0.95. One low reading
        can be the fly wobbling. Two is the setup having moved.
        """
        for p in self.positions:
            if p.symbol == symbol:
                if verdict_today < p.verdict - NOISE_FLOOR:
                    p.cooling += 1
                else:
                    p.cooling = 0
                return p.cooling >= 2
        return False

    def sell(self, symbol: str) -> OpenPosition | None:
        """Move a holding to pending. It is credited when the result arrives."""
        for i, p in enumerate(self.positions):
            if p.symbol == symbol:
                self.pending.append(self.positions.pop(i))
                return p
        return None

    def held(self) -> set[str]:
        return {p.symbol for p in self.positions}

    def drift(self) -> float:
        """How far the fly has moved from the connectome it started with."""
        moved = (np.abs(self.to_reward - self.baseline_reward)
                 + np.abs(self.to_punish - self.baseline_punish))
        return float(moved.mean())


def from_connectome(kc_body_ids: list[int], fly: str = "001") -> FlyMemory:
    """Build a fresh, untrained memory from the wiring.

    An output neuron's valence is read off which dopamine population drives
    it, so nothing here is assigned by hand.
    """
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    types = ann["type"].fillna("")
    mbon = set(ann[types.str.match(r"MBON")]["bodyId"].astype(int))
    pam = set(ann[types.str.match(r"PAM")]["bodyId"].astype(int))
    ppl = set(ann[types.str.match(r"PPL1")]["bodyId"].astype(int))

    w = pd.read_feather(ROOT / "data/malecns/weights.feather",
                        columns=["body_pre", "body_post", "weight"])
    onto_mbon = w[w.body_post.isin(mbon)]
    reward_side, punish_side = set(), set()
    for body in mbon:
        e = onto_mbon[onto_mbon.body_post == body]
        r = e[e.body_pre.isin(pam)].weight.sum()
        p = e[e.body_pre.isin(ppl)].weight.sum()
        if r > p:
            reward_side.add(body)
        elif p > r:
            punish_side.add(body)

    position = {b: i for i, b in enumerate(kc_body_ids)}
    to_reward = np.zeros(len(kc_body_ids))
    to_punish = np.zeros(len(kc_body_ids))
    for row in onto_mbon.itertuples():
        i = position.get(int(row.body_pre))
        if i is None:
            continue
        if int(row.body_post) in reward_side:
            to_reward[i] += row.weight
        elif int(row.body_post) in punish_side:
            to_punish[i] += row.weight
    scale = max(to_reward.max(), to_punish.max(), 1.0)
    return FlyMemory(to_reward / scale, to_punish / scale, kc_body_ids, FLIES / fly)
