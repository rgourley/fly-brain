"""Find the synaptic gain that gives MaleCNS the same resting activity as FlyWire.

w_syn was fitted on FlyWire. MaleCNS carries about twice the synaptic input
per neuron, so the same gain leaves the network saturated by its own
recurrent activity. This script measures spontaneous activity with no
stimulus on both datasets and sweeps w_syn on MaleCNS to match.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MALECNS = (ROOT / "data/malecns/2026_Completeness_malecns.csv",
           ROOT / "data/malecns/2026_Connectivity_malecns.parquet")
FLYWIRE = (ROOT / "data/2025_Completeness_783.csv",
           ROOT / "data/2025_Connectivity_783.parquet")
GAINS = ["0.275", "0.20", "0.145", "0.11", "0.08"]


def run(label: str, comp: Path, conn: Path, w_syn: str) -> tuple[int, int]:
    env = dict(os.environ)
    env["FLY_COMP_PATH"] = str(comp)
    env["FLY_CONN_PATH"] = str(conn)
    env["FLY_NEU_EXC"] = ""
    env["FLY_W_SYN"] = w_syn
    subprocess.run(
        [sys.executable, "main.py", "--brian2-cpu", "--experiment", "custom",
         "--t_run", "1", "--n_run", "1", "--run-label", label],
        cwd=ROOT, env=env, capture_output=True, check=True,
    )
    files = sorted((ROOT / "data/results" / label / "round_01").glob("*.parquet"))
    if not files:
        return 0, 0
    sp = pd.read_parquet(files[0])
    return len(sp), sp["flywire_id"].nunique()


def main() -> None:
    rows = []
    spikes, active = run("gain_flywire", *FLYWIRE, "0.275")
    rows.append({"dataset": "FlyWire", "w_syn": 0.275, "spikes": spikes,
                 "active": active, "pct_active": round(active / 138639 * 100, 2)})
    print(f"FlyWire  w_syn=0.275  spikes={spikes:>8,}  active={active:>7,}", flush=True)

    for g in GAINS:
        spikes, active = run(f"gain_mc_{g}", *MALECNS, g)
        rows.append({"dataset": "MaleCNS", "w_syn": float(g), "spikes": spikes,
                     "active": active, "pct_active": round(active / 164587 * 100, 2)})
        print(f"MaleCNS  w_syn={g:<6s} spikes={spikes:>8,}  active={active:>7,}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/gain_calibration.csv", index=False)
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
