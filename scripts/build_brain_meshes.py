"""Fetch MaleCNS ROI meshes and pack them for the concept page.

Source: the public release bucket gs://flyem-male-cns (CC BY 4.0, FlyEM at
HHMI Janelia, Google Research, Cambridge Connectomics Group). The meshes are
neuroglancer legacy format: uint32 vertex count, float32 xyz per vertex in
nanometres, uint32 triangle indices.

Both hemispheres are packed, because the simulation runs both. Each mesh is decimated by vertex clustering to a small triangle budget and
written into one binary file with a JSON manifest. Coordinates are recentred
on the whole set and scaled to micrometres.
"""

import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/concept/model/brain"
BUCKET = "https://storage.googleapis.com/flyem-male-cns/rois"
SUB = "malecns-subcompartments-v3"
FULL = "fullbrain-roi-v5"
SHELL = "brain-shell-v2.2"

GLOMERULI = ["D", "DA1", "DA2", "DA3", "DA4l", "DA4m", "DC1", "DC2", "DC3", "DC4",
             "DL1", "DL2d", "DL2v", "DL3", "DL4", "DL5", "DM1", "DM2", "DM3", "DM4",
             "DM5", "DM6", "DP1l", "DP1m", "V", "VA1d", "VA1v", "VA2", "VA3", "VA4",
             "VA5", "VA6", "VA7l", "VA7m", "VC1", "VC2", "VC3", "VC4", "VC5", "VL1",
             "VL2a", "VL2p", "VM1", "VM2", "VM3", "VM4", "VM5d", "VM5v", "VM6",
             "VM7d", "VM7v", "VP1d", "VP1l", "VP1m", "VP2", "VP3", "VP4", "VP5"]
COMPARTMENTS = ["a'1", "a'2", "a'3", "a1", "a2", "a3", "b'1", "b'2", "b1", "b2",
                "g1", "g2", "g3", "g4", "g5"]

# (name on the page, dataset, fragment name, triangle budget, group)
WANT = ([(f"AL-{g}({side})", SUB, f"AL-{g}({side})", 700, "glomerulus") for side in "RL" for g in GLOMERULI]
        + [(f"{c}({side})", SUB, f"{c}({side})", 1200, "compartment") for side in "RL" for c in COMPARTMENTS]
        + [(f"CA({side})", FULL, f"CA({side})", 2500, "calyx") for side in "RL"]
        + [(f"PED({side})", FULL, f"PED({side})", 1600, "pedunculus") for side in "RL"]
        + [("LH(R)", FULL, "LH(R)", 1200, "context"),
           ("LH(L)", FULL, "LH(L)", 1200, "context"),
           ("brain-shell", SHELL, "brain-shell", 12000, "shell")])


def fetch(dataset: str, fragment: str) -> tuple[np.ndarray, np.ndarray]:
    url = f"{BUCKET}/{dataset}/mesh/{quote(fragment)}.ngmesh"
    raw = urlopen(url, timeout=120).read()
    n = struct.unpack_from("<I", raw, 0)[0]
    verts = np.frombuffer(raw, dtype="<f4", count=n * 3, offset=4).reshape(-1, 3)
    tris = np.frombuffer(raw, dtype="<u4", offset=4 + n * 12).reshape(-1, 3)
    return verts.astype(np.float64), tris.astype(np.int64)


def decimate(verts: np.ndarray, tris: np.ndarray, budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering on a cubic grid, sized by bisection to hit the budget."""
    lo, hi = verts.min(0), verts.max(0)
    span = float((hi - lo).max())
    cell_lo, cell_hi = span / 400, span / 2
    best = None
    for _ in range(18):
        cell = (cell_lo + cell_hi) / 2
        key = np.floor((verts - lo) / cell).astype(np.int64)
        flat = key[:, 0] * 1_000_003 + key[:, 1] * 1_009 + key[:, 2]
        uniq, inv = np.unique(flat, return_inverse=True)
        pos = np.zeros((len(uniq), 3)); cnt = np.zeros(len(uniq))
        np.add.at(pos, inv, verts); np.add.at(cnt, inv, 1)
        pos /= cnt[:, None]
        t = inv[tris]
        t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
        t = np.unique(np.sort(t, axis=1), axis=0)
        if len(t) > budget:
            cell_lo = cell
        else:
            cell_hi = cell
            best = (pos, t)
    if best is None:
        best = (pos, t)
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        fetched = list(pool.map(lambda w: fetch(w[1], w[2]), WANT))
    meshes = []
    for (name, _, _, budget, group), (verts, tris) in zip(WANT, fetched):
        pos, t = decimate(verts, tris, budget)
        meshes.append((name, group, pos, t))
        print(f"{name:14s} {len(tris):8d} -> {len(t):6d} tris", flush=True)

    allpos = np.concatenate([m[2] for m in meshes])
    centre = (allpos.min(0) + allpos.max(0)) / 2
    manifest = []
    blob = bytearray()
    for name, group, pos, t in meshes:
        p = ((pos - centre) / 1000.0).astype("<f4")
        idx = t.astype("<u4")
        manifest.append({"name": name, "group": group, "offset": len(blob),
                         "vertices": int(len(p)), "triangles": int(len(idx)),
                         "centre": [round(float(v), 2) for v in p.mean(0)]})
        blob += p.tobytes() + idx.tobytes()
    (OUT / "brain.bin").write_bytes(blob)
    (OUT / "brain.json").write_text(json.dumps({
        "units": "micrometres, recentred",
        "source": "gs://flyem-male-cns/rois (MaleCNS v1.0, CC BY 4.0)",
        "meshes": manifest}, indent=1))
    print(f"wrote {len(blob) / 1e6:.2f} MB")


if __name__ == "__main__":
    sys.exit(main())
