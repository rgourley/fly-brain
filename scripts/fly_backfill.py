"""Upload replays that were saved before uploading was switched on.

    python scripts/fly_backfill.py            # every fly with a key, every saved replay
    python scripts/fly_backfill.py --fly 002

Uploading the same session twice replaces the row, so this is safe to run again.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_live import FLIES, api, keychain, load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default=None)
    args = ap.parse_args()
    for fly, row in load_manifest().items():
        if args.fly and fly != args.fly:
            continue
        key = keychain(row["keychain"])
        if not key:
            print(f"fly {fly}: no key in the keychain, skipped"); continue
        index = FLIES / fly / "replays" / "index.json"
        for name in json.loads(index.read_text()) if index.exists() else []:
            replay = json.loads((FLIES / fly / "replays" / name).read_text())
            # Replays saved before seconds were recorded carry a minute-precision time. The endpoint wants seconds.
            if len(replay["when"]) == len("2026-09-19T22:52+00:00"):
                replay["when"] = replay["when"][:16] + ":00" + replay["when"][16:]
            sent = api(key, "POST", "/fly/replays", json_body=replay)
            print(f"fly {fly}: {name} -> {sent['ran_at']}" + (" (rehearsal)" if sent.get("dry") else ""))
            time.sleep(1)


if __name__ == "__main__":
    main()
