"""Make the room backdrop for the concept page from the Poly Haven HDRI (CC0).

The 1k panorama looks blocky when shown sharp behind a 3 mm subject. A real
lens focused that close would throw the room far out of focus, so the
backdrop is the same panorama, tone-mapped and blurred, saved as a JPEG and
wrapped on a large sphere. Output: docs/concept/textures/room_backdrop.jpg
"""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/concept/textures/lythwood_room_1k.hdr"
OUT = ROOT / "docs/concept/textures/room_backdrop.jpg"


def read_hdr(path: Path) -> np.ndarray:
    """Radiance .hdr with new-style run-length encoding. Returns float RGB, shape (h, w, 3)."""
    data = path.read_bytes()
    head_end = data.index(b"\n\n") + 2
    line_end = data.index(b"\n", head_end)
    parts = data[head_end:line_end].split()
    h, w = int(parts[1]), int(parts[3])
    buf = np.frombuffer(data, dtype=np.uint8, offset=line_end + 1)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    p = 0
    for y in range(h):
        if not (buf[p] == 2 and buf[p + 1] == 2 and ((int(buf[p + 2]) << 8) | int(buf[p + 3])) == w):
            raise ValueError("scanline is not run-length encoded")
        p += 4
        for c in range(4):
            x = 0
            while x < w:
                n = int(buf[p]); p += 1
                if n > 128:
                    n -= 128; out[y, x:x + n, c] = buf[p]; p += 1
                else:
                    out[y, x:x + n, c] = buf[p:p + n]; p += n
                x += n
    scale = np.ldexp(1.0, out[..., 3].astype(np.int32) - 136)
    return out[..., :3].astype(np.float64) * scale[..., None]


def main() -> None:
    rgb = read_hdr(SRC)
    # Blur in linear light so bright windows bloom the way they do through a lens, wrapping at the seam.
    # The camera's field of view takes in about a twelfth of the panorama, so a heavy blur leaves a
    # flat wash. This much keeps the window and the lamps readable as soft shapes.
    sigma = (rgb.shape[0] / 170, rgb.shape[1] / 340)
    blurred = np.stack([gaussian_filter(rgb[..., c], sigma=sigma, mode=("nearest", "wrap")) for c in range(3)], axis=-1)
    exposed = blurred * 0.8
    mapped = exposed / (1.0 + exposed)                     # Reinhard, enough for a backdrop
    srgb = np.clip(mapped, 0, 1) ** (1 / 2.2)
    img = Image.fromarray((srgb * 255).astype(np.uint8)).resize((2048, 1024), Image.LANCZOS)
    img.save(OUT, quality=88)
    print(f"wrote {OUT.name}: {OUT.stat().st_size / 1e3:.0f} KB, from {rgb.shape[1]}x{rgb.shape[0]}")


if __name__ == "__main__":
    main()
