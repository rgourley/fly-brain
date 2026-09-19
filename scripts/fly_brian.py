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
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""


class BrianFly:
    """A loaded connectome that can be shown an animation.

    Loading the connectivity is the slow part, so build one and reuse it.
    """

    def __init__(self) -> None:
        comp = pd.read_csv(COMP, index_col=0)
        self.body_ids = list(comp.index)
        self.index_of = {b: i for i, b in enumerate(self.body_ids)}
        self.n = len(self.body_ids)
        con = pd.read_parquet(
            CONN,
            columns=["Presynaptic_Index", "Postsynaptic_Index",
                     "Excitatory x Connectivity"],
        )
        self.pre = con["Presynaptic_Index"].to_numpy()
        self.post = con["Postsynaptic_Index"].to_numpy()
        self.weight = con["Excitatory x Connectivity"].to_numpy()

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
        neu = NeuronGroup(self.n, EQS, method="linear",
                          threshold="v > v_th", reset="v = v_rst; g = 0 * mV",
                          refractory="rfc", namespace=PARAMS)
        neu.v = PARAMS["v_0"]
        neu.g = 0 * mV
        neu.rfc = PARAMS["t_rfc"]
        neu.rfc[stim_index] = 0 * ms

        syn = Synapses(neu, neu, "w : volt", on_pre="g += w",
                       delay=PARAMS["t_dly"])
        syn.connect(i=self.pre, j=self.post)
        syn.w = self.weight * PARAMS["w_syn"]

        drive = PoissonGroup(len(stim_ids), rates="stim(t, i)",
                             namespace={"stim": stim})
        feed = Synapses(drive, neu, on_pre="v_post += w_stim",
                        namespace={"w_stim": PARAMS["w_syn"] * PARAMS["f_poi"]})
        feed.connect(i=np.arange(len(stim_ids)), j=stim_index)

        monitor = SpikeMonitor(neu, record=False)
        net = Network(neu, syn, drive, feed, monitor)
        net.run(len(frames) * ms_per_frame * ms)

        return np.asarray(monitor.count[:], dtype=np.int64)
