"""Convert MaleCNS v1.0 flat-connectome files to the Shiu model data format.

The Shiu model reads two files:
  - a completeness CSV, indexed by body ID, that defines the neuron order
  - a connectivity parquet with pre/post IDs, row indices, synapse counts,
    and a +1/-1 excitatory sign

This script writes both from the MaleCNS weights and neurotransmitter tables.
Acetylcholine is excitatory. GABA and glutamate are inhibitory.
"""

import pandas as pd
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "data" / "malecns"
INHIBITORY = {"gaba", "glutamate"}

# Glutamate is inhibitory across most of the fly brain, which is why the
# blanket rule is standard. It is wrong in the visual system. L1 is
# glutamatergic and excites Mi1, and L1 to Mi1 to T4 is the ON motion
# pathway. Left on the default rule, L1 delivers 142,185 inhibitory
# synapses to Mi1, Mi1 never fires, and every motion detector is silent.
# These types are forced excitatory on the published physiology.
FORCE_EXCITATORY = {"L1"}


def main() -> None:
    weights = pd.read_feather(SRC / "weights.feather",
                              columns=["body_pre", "body_post", "weight"])
    print(f"edges: {len(weights):,}")

    bodies = pd.Index(
        sorted(set(weights["body_pre"]) | set(weights["body_post"])),
        name="body",
    )
    print(f"neurons: {len(bodies):,}")

    nt = pd.read_feather(SRC / "nt.feather", columns=["body", "consensus_nt"])
    nt = nt.drop_duplicates("body").set_index("body")["consensus_nt"]
    nt = nt.reindex(bodies)

    sign = pd.Series(1, index=bodies, dtype="int8")
    sign[nt.isin(INHIBITORY).values] = -1

    ann = pd.read_feather(SRC / "body-annotations.feather").drop_duplicates("bodyId")
    forced = ann[ann["type"].isin(FORCE_EXCITATORY)]["bodyId"]
    forced = bodies.intersection(pd.Index(forced))
    sign[forced] = 1
    print(f"forced excitatory: {len(forced):,} cells of type {sorted(FORCE_EXCITATORY)}")

    missing = int(nt.isna().sum())
    print(f"inhibitory: {int((sign < 0).sum()):,} | "
          f"excitatory: {int((sign > 0).sum()):,} | "
          f"no NT call (defaulted excitatory): {missing:,}")

    index_of = pd.Series(range(len(bodies)), index=bodies)
    out = pd.DataFrame({
        "Presynaptic_ID": weights["body_pre"].to_numpy(),
        "Postsynaptic_ID": weights["body_post"].to_numpy(),
        "Presynaptic_Index": index_of.reindex(weights["body_pre"]).to_numpy(),
        "Postsynaptic_Index": index_of.reindex(weights["body_post"]).to_numpy(),
        "Connectivity": weights["weight"].to_numpy(),
        "Excitatory": sign.reindex(weights["body_pre"]).to_numpy(),
    })
    out["Excitatory x Connectivity"] = out["Excitatory"] * out["Connectivity"]

    conn_path = SRC / "2026_Connectivity_malecns.parquet"
    comp_path = SRC / "2026_Completeness_malecns.csv"
    out.to_parquet(conn_path, index=False)
    pd.DataFrame({"Completed": True}, index=bodies).to_csv(comp_path)
    print(f"wrote {conn_path.name} and {comp_path.name}")


if __name__ == "__main__":
    main()
