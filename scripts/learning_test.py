"""Can the fly's mushroom body tell profitable setups from unprofitable ones?

Two hundred setups, labelled by a rule the fly is never told. Each becomes
a smell, the smell becomes a Kenyon cell pattern, and the question is
whether that pattern carries enough to separate the two classes on setups
held out of training.

Three conditions, and the controls are the point:

  fly         the real connectome
  shuffled    the same patterns with cells scrambled per setup, so the
              information survives but the wiring does not
  random      a random sparse expansion of the same inputs at the same
              sparsity, which is what the mushroom body is often said to
              be. If this matches the fly, the specific wiring adds nothing.

This uses logistic regression rather than the biological plasticity rule,
so it measures whether the information is present at all. The real rule is
a weaker learner, so these numbers are a ceiling rather than a forecast.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_smell import channel_names, load_channel_map, smell, to_rates
from fly_brian import BrianFly

N_SETUPS = 200
MS = 100.0


def make_setups(n: int, rng: np.random.Generator) -> list[dict]:
    """Realistic indicator readings, drawn independently."""
    setups = []
    for _ in range(n):
        setups.append({
            "rsi": float(np.clip(rng.normal(52, 16), 5, 95)),
            "bb_position": float(np.clip(rng.beta(2, 2), 0, 1)),
            "distance_from_sma50": float(rng.normal(0.01, 0.07)),
            "volume_ratio": float(np.clip(rng.lognormal(0.1, 0.45), 0.3, 4.0)),
            "price_change_5d": float(rng.normal(0.0, 0.05)),
            "rsi_trend": str(rng.choice(["rising", "falling", "flat"])),
        })
    return setups


def label(setup: dict) -> int:
    """The rule the fly has to discover: oversold, on volume, off its lows.

    Deliberately needs three readings together. A single indicator will not
    separate the classes, so the representation has to carry combinations.
    """
    score = ((setup["rsi"] < 45) + (setup["volume_ratio"] > 1.3)
             + (setup["bb_position"] < 0.4))
    return int(score >= 2)


def evaluate(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=0.05))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


def main() -> None:
    rng = np.random.default_rng(17)
    setups = make_setups(N_SETUPS, rng)
    y = np.array([label(s) for s in setups])
    print(f"{N_SETUPS} setups, {y.sum()} labelled profitable, {len(y) - y.sum()} not")

    channels = channel_names()
    channel_map = load_channel_map()
    smells = [smell(s) for s in setups]
    inputs = np.array([[a.get(c, 0.0) for c in channels] for a in smells])

    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    kc_ids = [int(b) for b in ann[ann["type"].fillna("").str.match(r"KC")]["bodyId"]]
    fly = BrianFly()
    kc_index = [fly.index_of[b] for b in kc_ids if b in fly.index_of]

    patterns = []
    for i, a in enumerate(smells):
        counts = fly.show_sequence([to_rates(a, channel_map)], ms_per_frame=MS, seed=1000 + i)
        patterns.append(counts[kc_index].astype(float))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_SETUPS}", flush=True)
    X = np.array(patterns)
    print(f"Kenyon cells active per setup: {(X > 0).mean() * 100:.1f}%")

    # Control one: same numbers, cells scrambled. Information intact, wiring gone.
    shuffled = X.copy()
    for i in range(len(shuffled)):
        np.random.default_rng(i).shuffle(shuffled[i])

    # Control two: a random sparse expansion of the same inputs, matched for
    # size and sparsity. This is the mushroom body as a generic random
    # projection rather than this particular animal's wiring.
    proj_rng = np.random.default_rng(99)
    weights = np.zeros((len(channels), X.shape[1]))
    for j in range(X.shape[1]):
        picks = proj_rng.choice(len(channels), size=6, replace=False)
        weights[picks, j] = 1.0
    raw = inputs @ weights
    keep = int((X > 0).mean() * X.shape[1])
    random_proj = np.zeros_like(raw)
    for i in range(len(raw)):
        top = np.argsort(raw[i])[-keep:]
        random_proj[i, top] = raw[i, top]

    print()
    for name, data in (("fly", X), ("shuffled", shuffled), ("random", random_proj)):
        m, s = evaluate(data, y)
        print(f"{name:10s} held-out AUC {m:.3f} +/- {s:.3f}")
    m, s = evaluate(inputs, y)
    print(f"{'raw input':10s} held-out AUC {m:.3f} +/- {s:.3f}   (the smell itself, no fly)")
    print("chance     0.500")


if __name__ == "__main__":
    main()
