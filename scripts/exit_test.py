"""Is the sell signal real, or is the fly just wobbling?

Two checks. A stock that recovers from oversold to overbought should make
the verdict fall steadily, because the smell genuinely changed. A stock
whose readings do not move should hold its verdict inside the run-to-run
noise, or the fly would sell for no reason.
"""
import sys; sys.path.insert(0,'/Users/rob/code/personal/fly-brain/scripts')
import numpy as np, pandas as pd
from fly_smell import load_channel_map, smell, to_rates
from fly_brian import BrianFly
from fly_memory import from_connectome

ROOT='/Users/rob/code/personal/fly-brain'
ann = pd.read_feather(f'{ROOT}/data/malecns/body-annotations.feather').drop_duplicates('bodyId')
kc = [int(b) for b in ann[ann['type'].fillna('').str.match(r'KC')]['bodyId']]
fly = BrianFly(); idx = [fly.index_of[b] for b in kc if b in fly.index_of]
m = from_connectome([fly.body_ids[i] for i in idx])
cm = load_channel_map()
TRIALS = 5

def verdicts(reading, tag):
    vs = []
    for t in range(TRIALS):
        c = fly.show_sequence([to_rates(smell(reading), cm)], ms_per_frame=50.0, seed=900+t)
        vs.append(m.verdict(c[idx] > 0))
    return float(np.mean(vs)), float(np.std(vs))

# A stock recovering, day by day, from oversold to overbought.
recovery = [
    ('day 0  bought here', dict(rsi=24, bb_position=0.08, distance_from_sma50=-0.12, volume_ratio=2.4, price_change_5d=-0.09, rsi_trend='falling')),
    ('day 2  bouncing   ', dict(rsi=36, bb_position=0.28, distance_from_sma50=-0.07, volume_ratio=1.8, price_change_5d=0.02, rsi_trend='rising')),
    ('day 5  recovered  ', dict(rsi=52, bb_position=0.52, distance_from_sma50=0.00, volume_ratio=1.3, price_change_5d=0.06, rsi_trend='rising')),
    ('day 9  extended   ', dict(rsi=68, bb_position=0.80, distance_from_sma50=0.07, volume_ratio=1.1, price_change_5d=0.09, rsi_trend='rising')),
    ('day 12 overbought ', dict(rsi=79, bb_position=0.94, distance_from_sma50=0.12, volume_ratio=1.0, price_change_5d=0.11, rsi_trend='rising')),
]
print('A stock recovering: does the verdict fall?')
base = None
for tag, r in recovery:
    mu, sd = verdicts(r, tag)
    if base is None: base = mu
    print(f'  {tag}  verdict {mu:+.3f} +/- {sd:.3f}   change from purchase {mu-base:+.3f}')

print()
print('The same stock, unchanged, five days running: does it hold still?')
flat = recovery[0][1]
seen = []
for day in range(5):
    vs = []
    for t in range(TRIALS):
        c = fly.show_sequence([to_rates(smell(flat), cm)], ms_per_frame=50.0, seed=2000+day*7+t)
        vs.append(m.verdict(c[idx] > 0))
    seen.append(float(np.mean(vs)))
    print(f'  day {day}  verdict {seen[-1]:+.3f}')
print(f'\n  drift across five unchanged days: {max(seen)-min(seen):+.3f}')
