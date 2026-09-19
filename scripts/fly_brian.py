"""Show the fly an animation using Brian2, which is fast enough to be useful.

The PyTorch backend takes a fresh rate per cell per timestep, which is what
an animation needs, but it multiplies the whole 25-million-connection matrix
ten thousand times per simulated second and takes roughly an hour per
decision. Brian2 compiles to C++ and only touches synapses that carry a
spike, which is several hundred times faster.

Brian2's usual stimulus holds one rate for a whole run. This drives it from
a timed array instead, so every cell can follow its own frame by frame.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from brian2 import (Hz, Network, NeuronGroup, PoissonGroup, SpikeMonitor,
                    Synapses, TimedArray, mV, ms, defaultclock, prefs, second)

ROOT = Path(__file__).resolve().parent.parent
COMP = ROOT / "data/malecns/2026_Completeness_malecns.csv"
CONN = ROOT / "data/malecns/2026_Connectivity_malecns.parquet"
ANN = ROOT / "data/malecns/body-annotations.feather"
WEIGHTS = ROOT / "data/malecns/weights.feather"

prefs.codegen.target = "cython"

PARAMS = {
    "v_0": -52 * mV,
    "v_rst": -52 * mV,
    "v_th": -45 * mV,
    "t_mbr": 20 * ms,
    "tau": 5 * ms,
    "t_rfc": 2.2 * ms,
    "t_dly": 1.8 * ms,
    "w_syn": 0.275 * mV,
    "f_poi": 250,
}

EQS = """
dv/dt = (v_0 - v + g - inhib) / t_mbr : volt (unless refractory)
dg/dt = -g / tau                       : volt (unless refractory)
dtrace/dt = -trace / t_apl             : 1
inhib                                  : volt
rfc                                    : second
"""

# APL does not spike. It releases GABA continuously, in proportion to how
# much Kenyon cell activity it sees, and that feedback is what holds the
# mushroom body at roughly 5% active. Forced to spike like every other cell
# in this model it fires at 363 Hz and holds nothing down, so odours all
# produce the same pattern and cannot be told apart.
#
# APL_GAIN is the one number here that is not from the connectome. It is set
# so sparseness matches the level measured in real flies. The per-cell
# strengths still come from the 196,200 APL synapses in the data.
# Gain 1000 puts the mushroom body at 3.9% active, matching the sparseness
# measured in real flies. Below it the cells saturate and every odour looks
# alike; above it the representation thins out and starts losing detail.
# Tuned against the sliced circuit. Gain 1000 was for the whole brain,
# which delivered far more drive through routes that are not the olfactory
# pathway. With those gone, 250 puts the mushroom body at 4.3% active.
APL_GAIN = 250.0
APL_PARAMS = {"t_apl": 100 * ms}


class BrianFly:
    """A loaded connectome that can be shown an animation.

    Loading the connectivity is the slow part, so build one and reuse it.
    """

    def __init__(self, apl_gain: float = APL_GAIN,
                 w_syn: float | None = None,
                 olfactory_only: bool = True) -> None:
        """Optionally weaken every synapse, or keep only the smell circuit.

        Lowering w_syn makes signals fade as they travel, so a four-hop
        indirect route arrives weaker than the one-hop direct one, which is
        what a real neuron does and what this model otherwise skips.

        olfactory_only keeps receptors, their direct targets, Kenyon cells,
        APL, the output neurons and the dopamine cells. That removes the
        indirect routes entirely rather than damping them.
        """
        self.w_syn = PARAMS["w_syn"] if w_syn is None else w_syn * mV
        self.olfactory_only = olfactory_only
        comp = pd.read_csv(COMP, index_col=0)
        self.body_ids = list(comp.index)
        self.index_of = {b: i for i, b in enumerate(self.body_ids)}
        self.n = len(self.body_ids)
        self.apl_gain = apl_gain
        self._load_mushroom_body()
        con = pd.read_parquet(
            CONN,
            columns=["Presynaptic_Index", "Postsynaptic_Index",
                     "Excitatory x Connectivity"],
        )
        self.pre = con["Presynaptic_Index"].to_numpy()
        self.post = con["Postsynaptic_Index"].to_numpy()
        self.weight = con["Excitatory x Connectivity"].to_numpy()
        if olfactory_only:
            self._restrict_to_smell_circuit()

    def _load_mushroom_body(self) -> None:
        """Find APL and the Kenyon cells, and read their synapse counts.

        The graded feedback uses the real per-cell strengths: how much each
        Kenyon cell drives APL, and how much APL inhibits each one.
        """
        ann = pd.read_feather(ANN).drop_duplicates("bodyId")
        types = ann["type"].fillna("")
        apl_ids = [int(b) for b in ann[types.str.contains("APL", na=False)]["bodyId"]]
        kc_ids = [int(b) for b in ann[types.str.match(r"KC")]["bodyId"]]
        self.apl = [self.index_of[b] for b in apl_ids if b in self.index_of]
        self.kc = [self.index_of[b] for b in kc_ids if b in self.index_of]

        weights = pd.read_feather(WEIGHTS,
                                  columns=["body_pre", "body_post", "weight"])
        apl_set, kc_set = set(apl_ids), set(kc_ids)
        to_apl = weights[weights.body_pre.isin(kc_set) & weights.body_post.isin(apl_set)]
        from_apl = weights[weights.body_pre.isin(apl_set) & weights.body_post.isin(kc_set)]
        self.kc_drive = to_apl.groupby("body_pre")["weight"].sum().to_dict()
        self.apl_strength = from_apl.groupby("body_post")["weight"].sum()
        # Normalise so the gain, not the raw synapse count, sets the scale.
        self.apl_strength = (self.apl_strength / self.apl_strength.mean()).to_dict()

    def _restrict_to_smell_circuit(self) -> None:
        """Drop every neuron that is not part of the olfactory pathway.

        Any single glomerulus reaches more than half the Kenyon cells in the
        whole brain, because activity finds its way round through unrelated
        regions. Those routes exist in the animal too but arrive far too
        weak to matter, and this model gives every synapse the same strength
        so they arrive as loud as the direct path. Removing them restores
        the pathway the biology actually uses.
        """
        ann = pd.read_feather(ANN).drop_duplicates("bodyId")
        types = ann["type"].fillna("")
        orn = set(int(b) for b in ann[ann["class"] == "olfactory"]["bodyId"])
        kc = set(int(b) for b in ann[types.str.match(r"KC")]["bodyId"])
        keep = set(orn) | set(kc)
        keep |= set(int(b) for b in ann[types.str.contains("APL", na=False)]["bodyId"])
        keep |= set(int(b) for b in ann[types.str.match(r"MBON")]["bodyId"])
        keep |= set(int(b) for b in ann[types.str.match(r"(PAM|PPL1)")]["bodyId"])

        # Projection neurons: whatever takes input from receptors and feeds
        # Kenyon cells. Defined by the wiring rather than by a name.
        w = pd.read_feather(WEIGHTS, columns=["body_pre", "body_post", "weight"])
        from_orn = set(w[w.body_pre.isin(orn)]["body_post"])
        to_kc = set(w[w.body_post.isin(kc)]["body_pre"])
        keep |= (from_orn & to_kc)

        allowed = np.array([self.body_ids[i] in keep for i in range(self.n)])
        mask = allowed[self.pre] & allowed[self.post]
        self.pre, self.post = self.pre[mask], self.post[mask]
        self.weight = self.weight[mask]
        self.kept = int(allowed.sum())

    def show_sequence(self, frames: list[dict[int, float]],
                      ms_per_frame: float = 25.0,
                      seed: int | None = None) -> np.ndarray:
        """Play frames of per-cell rates and return spike counts per neuron.

        Keys in each frame are body ids. Cells absent from a frame are dark
        for that frame. The network runs straight through, so motion between
        frames reaches the direction selective cells.
        """
        if seed is not None:
            np.random.seed(seed)

        stim_ids = sorted({b for frame in frames for b in frame})
        stim_index = np.array([self.index_of[b] for b in stim_ids])
        rates = np.zeros((len(frames), len(stim_ids)), dtype=np.float64)
        position = {b: k for k, b in enumerate(stim_ids)}
        for t, frame in enumerate(frames):
            for body_id, hz in frame.items():
                rates[t, position[body_id]] = hz

        stim = TimedArray(rates * Hz, dt=ms_per_frame * ms)

        defaultclock.dt = 0.1 * ms
        params = {**PARAMS, **APL_PARAMS}
        neu = NeuronGroup(self.n, EQS, method="euler",
                          threshold="v > v_th",
                          reset="v = v_rst; g = 0 * mV; trace += 1",
                          refractory="rfc", namespace=params)
        neu.v = PARAMS["v_0"]
        neu.g = 0 * mV
        neu.inhib = 0 * mV
        neu.trace = 0
        neu.rfc = PARAMS["t_rfc"]
        neu.rfc[stim_index] = 0 * ms

        syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                       delay=PARAMS["t_dly"])
        syn.connect(i=self.pre, j=self.post)
        syn.w = self.weight * self.w_syn

        drive = PoissonGroup(len(stim_ids), rates="stim(t, i)",
                             namespace={"stim": stim})
        feed = Synapses(drive, neu, on_pre="v_post += w_stim",
                        namespace={"w_stim": self.w_syn * PARAMS["f_poi"]})
        feed.connect(i=np.arange(len(stim_ids)), j=stim_index)

        objects = [neu, syn, drive, feed]
        if self.apl_gain > 0 and self.apl and self.kc:
            objects += self._graded_apl(neu)

        monitor = SpikeMonitor(neu, record=False)
        net = Network(*objects, monitor)
        net.run(len(frames) * ms_per_frame * ms)

        return np.asarray(monitor.count[:], dtype=np.int64)

    def _graded_apl(self, neu):
        """Replace APL's spiking output with continuous feedback inhibition.

        Kenyon cell activity sums into APL, and APL pushes back on every
        Kenyon cell in proportion to that total. This is negative feedback,
        so the more cells fire the harder they are damped, which is what
        leaves only the most strongly driven ones active.
        """
        apl = NeuronGroup(1, "level : 1", namespace={})
        gather = Synapses(neu, apl, "level_post = w_k * trace_pre : 1 (summed)\n"
                                    "w_k : 1")
        gather.connect(i=self.kc, j=0)
        gather.w_k = [self.kc_drive.get(self.body_ids[i], 0) for i in self.kc]
        gather.w_k = gather.w_k[:] / max(float(np.mean(gather.w_k[:])), 1e-9) / len(self.kc)

        spread = Synapses(apl, neu, "inhib_post = gain * w_a * level_pre * mV : volt (summed)\n"
                                    "w_a : 1",
                          namespace={"gain": self.apl_gain})
        spread.connect(i=0, j=self.kc)
        spread.w_a = [self.apl_strength.get(self.body_ids[i], 1.0) for i in self.kc]
        return [apl, gather, spread]
