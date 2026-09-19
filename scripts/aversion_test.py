"""How many losses before the fly actually dislikes a setup?

The verdict is reward-side drive minus punishment-side drive. Untrained it
sits positive for ordinary patterns, so the fly finds everything somewhat
appealing and dislike is only relative. Losses depress the reward side for
the cells that were firing, so enough of them should tip the balance.

Run with forgetting on as well, because in a real session the synapses drift
home between trades and that sets a floor on how far dislike can go.
"""
import sys; sys.path.insert(0,'/Users/rob/code/personal/fly-brain/scripts')
import numpy as np, pandas as pd
from fly_smell import load_channel_map, smell, to_rates
from fly_brian import BrianFly
from fly_memory import from_connectome, LEARNING_RATE, DECAY_RATE
from innate_test import SETUPS

ROOT='/Users/rob/code/personal/fly-brain'
ann = pd.read_feather(f'{ROOT}/data/malecns/body-annotations.feather').drop_duplicates('bodyId')
kc = [int(b) for b in ann[ann['type'].fillna('').str.match(r'KC')]['bodyId']]
fly = BrianFly()
idx = [fly.index_of[b] for b in kc if b in fly.index_of]
bodies = [fly.body_ids[i] for i in idx]
cm = load_channel_map()

chosen = ['oversold, heavy volume, near lows', 'overbought breakout', 'dead flat']
pats = {}
for name in chosen:
    c = fly.show_sequence([to_rates(smell(SETUPS[name]), cm)], ms_per_frame=50.0, seed=77)
    pats[name] = c[idx] > 0

print(f'learning rate {LEARNING_RATE}, decay {DECAY_RATE} per session\n')
for name in chosen:
    for decay_on in (False, True):
        m = from_connectome(bodies)
        cells = np.flatnonzero(pats[name])
        start = m.verdict(pats[name])
        flipped = None
        for n in range(1, 201):
            m.learn(cells, profitable=False)
            if decay_on:
                m.forget()
            v = m.verdict(pats[name])
            if v < 0 and flipped is None:
                flipped = n
                break
        end = m.verdict(pats[name])
        tag = 'with forgetting' if decay_on else 'no forgetting  '
        got = f'{flipped} losses' if flipped else f'never (settles at {end:+.2f})'
        print(f'{name[:30]:32s} {tag}  start {start:+.2f}  -> {got}')
