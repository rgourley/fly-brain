"""Run the fly on a set of per-cell firing rates and return what fired.

The Brian2 harness drives every stimulated cell at one rate, which cannot
represent an image. This runs the same model on PyTorch, where each cell
gets its own rate, and hands back spike counts per neuron.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from run_pytorch import MODEL_PARAMS, DT, TorchModel, get_hash_tables, get_weights

COMP = ROOT / "data/malecns/2026_Completeness_malecns.csv"
CONN = ROOT / "data/malecns/2026_Connectivity_malecns.parquet"


class Fly:
    """A loaded connectome that can be shown things.

    Loading takes a while, so build one and reuse it across presentations.
    """

    def __init__(self, trials: int = 8, device: str = "cpu") -> None:
        self.flyid2i, _ = get_hash_tables(COMP)
        self.size = len(self.flyid2i)
        self.trials = trials
        self.device = device
        weights = get_weights(CONN, COMP, ROOT / "data/malecns", csr=False)
        self.model = TorchModel(batch=trials, size=self.size, dt=DT,
                                params=MODEL_PARAMS, weights=weights,
                                device=device)

    def show(self, rates: dict[int, float], ms: float = 200.0,
             seed: int | None = None) -> np.ndarray:
        """Present one still stimulus and return spike counts per neuron."""
        return self.show_sequence([rates], ms_per_frame=ms, seed=seed)

    def show_sequence(self, frames: list[dict[int, float]],
                      ms_per_frame: float = 25.0,
                      seed: int | None = None) -> np.ndarray:
        """Play a sequence of frames and return spike counts per neuron.

        The network is not reset between frames, so motion across frames
        reaches the direction-selective cells. Every trial sees the same
        frames. They differ because the input is Poisson, which is where
        the fly's own variability comes from.
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        state = self.model.state_init()
        counts = torch.zeros((self.trials, self.size), dtype=torch.int32,
                             device=self.device)
        rate_vector = torch.zeros((self.trials, self.size), dtype=torch.float32,
                                  device=self.device)
        steps = int(ms_per_frame / DT)

        for rates in frames:
            rate_vector.zero_()
            for body_id, hz in rates.items():
                idx = self.flyid2i.get(body_id)
                if idx is not None:
                    rate_vector[:, idx] = hz
            for _ in range(steps):
                state = self.model(rate_vector, *state, generator=generator)
                counts += state[2].to(torch.int32)
        return counts.cpu().numpy()


def region_response(counts: np.ndarray, flyid2i: dict[int, int],
                    readout: pd.DataFrame) -> pd.DataFrame:
    """Total the readout neurons' spikes, split by which region they favour.

    `readout` needs a body_id column and a region column.
    """
    rows = []
    for region, group in readout.groupby("region"):
        idx = [flyid2i[b] for b in group["body_id"] if b in flyid2i]
        if not idx:
            continue
        per_trial = counts[:, idx].sum(axis=1)
        rows.append({"region": int(region),
                     "mean": float(per_trial.mean()),
                     "std": float(per_trial.std()),
                     "n_cells": len(idx)})
    return pd.DataFrame(rows).sort_values("mean", ascending=False)
