"""How much of the ceiling does the fly's own learning rule reach?

The representation carries enough to separate good setups from bad at 0.960
when read by logistic regression. That is an upper bound. This measures
what dopamine-gated plasticity gets, which is a far weaker learner and the
one the animal actually has.

The rule: reward dopamine depresses the Kenyon cell synapses onto the
output neurons in its own compartment. Profit depresses the synapses onto
the punishment-driven outputs, so avoidance falls and the verdict rises.
Loss does the reverse. Only the cells that were firing get changed, which
is what ties the lesson to that particular smell.

Scoring is prequential: every setup is predicted before it is learned from,
so nothing is ever tested on something it has already seen. That matches
how the agent would actually run.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_smell import channel_names, load_channel_map, smell, to_rates
from fly_brian import BrianFly
from learning_test import N_SETUPS, make_setups, label

CACHE = ROOT / "data" / "malecns" / "kc_patterns.npz"
LEARNING_RATE = 0.02
PASSES = 6


def kenyon_patterns() -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Kenyon cell activity for each setup, cached because it is slow."""
    rng = np.random.default_rng(17)
    setups = make_setups(N_SETUPS, rng)
    y = np.array([label(s) for s in setups])

    if CACHE.exists():
        return np.load(CACHE)["patterns"], y, setups

    channel_map = load_channel_map()
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    kc_ids = [int(b) for b in ann[ann["type"].fillna("").str.match(r"KC")]["bodyId"]]
    fly = BrianFly()
    index = [fly.index_of[b] for b in kc_ids if b in fly.index_of]

    rows = []
    for i, s in enumerate(setups):
        counts = fly.show_sequence([to_rates(smell(s), channel_map)],
                                   ms_per_frame=100.0, seed=1000 + i)
        rows.append(counts[index].astype(float))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_SETUPS}", flush=True)
    patterns = np.array(rows)
    np.savez_compressed(CACHE, patterns=patterns,
                        body_ids=np.array([fly.body_ids[i] for i in index]))
    return patterns, y, setups


def output_groups() -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Split the output neurons by which dopamine population drives them.

    Reward cells and punishment cells target different compartments, so an
    output neuron's valence comes out of the wiring rather than from us.
    """
    ann = pd.read_feather(ROOT / "data/malecns/body-annotations.feather").drop_duplicates("bodyId")
    types = ann["type"].fillna("")
    mbon = ann[types.str.match(r"MBON")]["bodyId"].astype(int).tolist()
    pam = set(ann[types.str.match(r"PAM")]["bodyId"].astype(int))
    ppl = set(ann[types.str.match(r"PPL1")]["bodyId"].astype(int))

    w = pd.read_feather(ROOT / "data/malecns/weights.feather",
                        columns=["body_pre", "body_post", "weight"])
    onto = w[w.body_post.isin(set(mbon))]
    reward, punish = [], []
    for body in mbon:
        e = onto[onto.body_post == body]
        r = e[e.body_pre.isin(pam)].weight.sum()
        p = e[e.body_pre.isin(ppl)].weight.sum()
        if r > p:
            reward.append(body)
        elif p > r:
            punish.append(body)

    kc_to_mbon = w[w.body_post.isin(set(mbon))]
    return reward, punish, kc_to_mbon, np.array(mbon)


def main() -> None:
    patterns, y, setups = kenyon_patterns()
    active = patterns > 0
    print(f"{len(y)} setups, {int(y.sum())} profitable, "
          f"{active.mean() * 100:.1f}% of Kenyon cells active each")

    reward, punish, kc_to_mbon, mbon = output_groups()
    print(f"output neurons: {len(reward)} reward-driven, {len(punish)} punishment-driven")

    body_ids = np.load(CACHE)["body_ids"]
    position = {int(b): i for i, b in enumerate(body_ids)}
    reward_set, punish_set = set(reward), set(punish)

    # Start from the connectome: how strongly each Kenyon cell drives the
    # reward side and the punishment side.
    to_reward = np.zeros(len(body_ids))
    to_punish = np.zeros(len(body_ids))
    for row in kc_to_mbon.itertuples():
        i = position.get(int(row.body_pre))
        if i is None:
            continue
        if int(row.body_post) in reward_set:
            to_reward[i] += row.weight
        elif int(row.body_post) in punish_set:
            to_punish[i] += row.weight
    scale = max(to_reward.max(), to_punish.max(), 1.0)
    to_reward /= scale
    to_punish /= scale
    start_r, start_p = to_reward.copy(), to_punish.copy()

    rng = np.random.default_rng(5)
    predictions, truths = [], []
    for _ in range(PASSES):
        for i in rng.permutation(len(y)):
            cells = active[i]
            verdict = float(to_reward[cells].sum() - to_punish[cells].sum())
            predictions.append(verdict)
            truths.append(int(y[i]))
            # Dopamine depresses the opposite compartment for active cells.
            if y[i] == 1:
                to_punish[cells] *= (1.0 - LEARNING_RATE)
            else:
                to_reward[cells] *= (1.0 - LEARNING_RATE)

    per_pass = len(y)
    print()
    for p in range(PASSES):
        lo, hi = p * per_pass, (p + 1) * per_pass
        auc = roc_auc_score(truths[lo:hi], predictions[lo:hi])
        print(f"  pass {p + 1}: AUC {auc:.3f}")

    baseline = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01)
    channels = channel_names()
    raw = np.array([[smell(s).get(c, 0.0) for c in channels] for s in setups])
    preds_b, truth_b = [], []
    rng2 = np.random.default_rng(5)
    for p in range(PASSES):
        for i in rng2.permutation(len(y)):
            if p > 0 or i != 0:
                try:
                    preds_b.append(float(baseline.decision_function(raw[i:i + 1])[0]))
                    truth_b.append(int(y[i]))
                except Exception:
                    pass
            baseline.partial_fit(raw[i:i + 1], y[i:i + 1], classes=np.array([0, 1]))
    tail = len(y)
    print()
    print(f"fly, final pass        AUC {roc_auc_score(truths[-tail:], predictions[-tail:]):.3f}")
    print(f"simple model, same way AUC {roc_auc_score(truth_b[-tail:], preds_b[-tail:]):.3f}")
    print(f"ceiling from before    AUC 0.960")
    moved = np.mean(np.abs(to_reward - start_r) + np.abs(to_punish - start_p))
    print(f"\nmean synaptic change: {moved:.4f}")


if __name__ == "__main__":
    main()
