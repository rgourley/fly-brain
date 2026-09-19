# Fly swarm: design and handoff

Everything needed to build this without the conversation it came from.

## What already exists

A single fly that works, in `scripts/`:

| File | Does |
|---|---|
| `convert_malecns.py` | Turns the MaleCNS release into the two files the Shiu model reads |
| `fly_brian.py` | Runs the olfactory circuit on Brian2, with graded APL |
| `fly_smell.py` | Turns six indicators into a smell across 51 glomeruli |
| `fly_memory.py` | Persistent synapses, open positions, learning, forgetting |
| `fly_agent.py` | One trading session end to end |

It smells a board, ranks it, buys, remembers between runs, credits closed
trades back to the cells that chose them, and can learn to avoid a setup
after seven to eleven losses.

## The swarm

One fly has one set of preferences. Twenty flies have twenty, and watching
them disagree is more interesting than watching one of them be confident.

### What differs between individuals

Only one thing: **which projection neurons feed which Kenyon cells.**

This is not a modelling choice. Kenyon cells sample their inputs randomly
while the fly develops, so the wiring differs in every animal, which is why
individual flies have stable individual odour preferences. Everything else
stays identical: same receptors, same antennal lobe, same APL, same output
neurons, same dopamine wiring, same rules.

Build each fly by reshuffling the PN-to-KC connections while preserving
each Kenyon cell's input count. Seed the shuffle per fly so it reproduces.

### Sizing

Each fly independently picks its favourite stock on the board. **Position
size is how many flies picked the same one.** Fifteen of twenty agreeing is
conviction. Votes scattered three and two and four across six names means
nobody is sure and little gets bought.

This replaces the margin-over-second-place rule in `fly_agent.size()`, which
was the weakest part of the single-fly design. Consensus from counting
independent opinions beats a formula.

### Outcomes are shared, memories are not

The swarm trades one portfolio. Every fly sees the same result and updates
its own synapses through its own wiring. They learn the same facts through
different lenses, so they stay different rather than converging into twenty
copies of each other.

One memory file per fly. `fly_memory.py` already takes a path; parameterise
it.

### Implementation: one network, not twenty

All flies share the front end. Do **not** run twenty simulations.

Build one Brian2 network per stock containing the shared receptors and
projection neurons, plus twenty separate Kenyon cell populations, each
wired with its own shuffle and its own APL. That is eight simulations for an
eight-stock board rather than one hundred and sixty, and it is exact rather
than an approximation because every fly's cells are genuinely simulated.

Roughly 80,000 Kenyon cells plus the shared front, which is smaller than the
whole-brain network this started with.

## Parameters we chose

These are not from the connectome. They are decisions and they belong on the
agent's public page, not buried.

| Parameter | Value | Set by |
|---|---|---|
| APL gain | 250 | Reproduces the ~4% Kenyon cell sparseness measured in real flies |
| Presentation | 50 ms | Shortest length that keeps the generalisation gradient while allowing exploration |
| Learning rate | 0.05 | One outcome should not swing an opinion |
| Decay rate | 0.02 per session | Lessons fade unless reconfirmed; also stops synapses walking to zero |
| Spikes to dollars | 250 per unit verdict | Arbitrary. Nothing biological sets this |

## Two corrections to the published model

Both are physiological facts the model's uniform-neuron assumption misses.
Both are in the code with reasoning attached.

**L1 is forced excitatory** (`convert_malecns.py`). Glutamate inhibits across
most of the fly brain, which is why the blanket rule is standard, but L1
excites Mi1 and that is the ON motion pathway. Left on the default rule, L1
delivers 142,185 inhibitory synapses to Mi1, Mi1 never fires, and every
motion detector in the brain is silent.

**APL releases continuously rather than spiking** (`fly_brian.py`). It is one
of the fly's non-spiking neurons. Forced to spike it fires at 363 Hz and
suppresses nothing, so every odour lights every Kenyon cell and no two smells
can be told apart.

## Run the circuit, not the whole brain

The single most important structural decision. Any one glomerulus reaches
2,215 of the 4,064 Kenyon cells in a whole-brain simulation, because activity
finds its way round through unrelated regions. Those routes exist in the
animal but arrive far too weak to matter. This model gives every synapse the
same strength, so a four-hop whisper arrives as loud as the one-hop shout.

Measured: two unrelated channels share 60% of their Kenyon cells in the whole
brain, and 0% in the sliced circuit.

Weakening every synapse does not fix it, because it damps the direct path as
much as the indirect ones. 49% overlap at best, then the network dies.

## What has already been ruled out

Do not spend time on these. Each was measured, not assumed.

| Idea | Result |
|---|---|
| Steering by odour direction | Dead. Olfactory neurons send 51.9% of output to each hemisphere, so side information is gone at the first synapse. Wind neurons keep 91.4% and do work |
| Detecting rising vs falling smell | Dead. 0.5% difference at the same concentration. No usable history in the network |
| Motion detection on charts | Dead. Direction selectivity is built from differential synaptic delays; every synapse here has the same 1.8 ms |
| Looming detection | Dead. LPLC2 reads from T4/T5, which are dead for the above reason |
| Reading chart shape visually | The optic lobe response correlates with total ink on the chart at r=0.975. It is a pixel counter |
| Vision into the learning circuit | Visual input is 0.53% of Kenyon cell input. Not worth the pipeline |

## The comparison that matters

On held-out setups: the fly scores 0.923, a random projection of the same
shape scores 0.924, and feeding the six indicators straight into a regression
scores 0.948.

**The connectome adds nothing over an arbitrary random expansion, and the
expansion loses a little of what was in the input.** This is the honest
headline and it should be on the page. What the fly has that a regression
does not is a mechanism that is real: it learns at the synapse flies actually
use, with a generalisation gradient that falls off with similarity the way a
real animal's does, and nothing fitted to market outcomes anywhere.

## The exit does not do what you would assume

A stock bought oversold at verdict +2.77 dips to +1.82 while it is still
messy, then climbs to +5.50 once it is overbought. The fly likes a recovering
position **more**, because untrained it innately prefers breakouts.

So "sell when the fly likes it less than when it bought" never fires on a
winner. It fires when a position goes quiet, since quiet drift is the setup
this fly likes least. The fly exits boredom, not strength.

That is trend following by disposition rather than by design, and nobody
chose it. It is also not a risk control. Any stop or drawdown limit is a rule
you are adding and should be labelled as yours rather than the animal's.

Run-to-run noise is 0.36 against real changes of about 0.95, so require two
consecutive sessions of a lower verdict before selling.

## Open items

- ClawStreet wiring: registration, orders with an idempotency key, asking the
  API which positions closed rather than being handed a dictionary
- Post the thought within five seconds of the order so they pair in the feed
- Attribution on the agent page: FlyEM at HHMI Janelia, the Cambridge
  Connectomics Group, and Google Research, under CC-BY
- Decide whether the innate preference driving trades is the raw verdict,
  which favours whichever setup shouts loudest, or the per-cell figure, which
  favours oversold setups. Both are measurements; picking one is a decision
