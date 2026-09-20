"""Install or remove the launchd jobs that run the flies on this Mac.

    python scripts/fly_schedule.py install --mode record   # rehearsals: real data, real brain, no orders
    python scripts/fly_schedule.py install --mode go       # live: places orders and posts thoughts
    python scripts/fly_schedule.py install --mode go --fly 002   # one fly only
    python scripts/fly_schedule.py remove
    python scripts/fly_schedule.py status

One job per fly in data/flies/flies.json. launchd calls the runner twice an hour,
on the hour and the half hour plus the fly's "minute_offset"; the runner's --if-due check decides whether a session is
due, so the timing rules live in one place (fly_live.slot) and survive
daylight saving. A Mac that is asleep runs nothing: launchd fires the missed
call once on wake, and --if-due drops it if the window has passed.
"""

import argparse
import json
import os
import plistlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLIES = ROOT / "data" / "flies"
AGENTS = Path.home() / "Library" / "LaunchAgents"
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def label(fly: str) -> str:
    return f"io.clawstreet.fly{fly}"


def domain() -> str:
    return f"gui/{os.getuid()}"


def unload(fly: str) -> None:
    subprocess.run(["launchctl", "bootout", f"{domain()}/{label(fly)}"], capture_output=True)


def install(mode: str, only: str | None = None) -> None:
    AGENTS.mkdir(parents=True, exist_ok=True)
    for fly, row in json.loads((FLIES / "flies.json").read_text()).items():
        if only and fly != only:
            continue
        # Two flies starting in the same minute load the circuit at once and both think slower,
        # which widens the gap between quote and order. "minute_offset" in the manifest spreads them out.
        offset = int(row.get("minute_offset", 0))
        (FLIES / fly).mkdir(parents=True, exist_ok=True)
        log = str(FLIES / fly / "run.log")
        plist = {
            "Label": label(fly),
            "ProgramArguments": [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/fly_live.py"),
                                 "--fly", fly, "--if-due", f"--{mode}"],
            "WorkingDirectory": str(ROOT),
            "StartCalendarInterval": [{"Minute": offset}, {"Minute": 30 + offset}],
            "EnvironmentVariables": {"PATH": PATH},
            "StandardOutPath": log,
            "StandardErrorPath": log,
            "ProcessType": "Background",
        }
        path = AGENTS / f"{label(fly)}.plist"
        unload(fly)
        path.write_bytes(plistlib.dumps(plist))
        subprocess.run(["launchctl", "bootstrap", domain(), str(path)], check=True)
        print(f"{label(fly)}: installed, mode {mode}, runs at :{offset:02d} and :{30 + offset:02d}, log {log}")


def remove() -> None:
    for fly in json.loads((FLIES / "flies.json").read_text()):
        unload(fly)
        (AGENTS / f"{label(fly)}.plist").unlink(missing_ok=True)
        print(f"{label(fly)}: removed")


def status() -> None:
    for fly in json.loads((FLIES / "flies.json").read_text()):
        path = AGENTS / f"{label(fly)}.plist"
        if not path.exists():
            print(f"{label(fly)}: not installed"); continue
        plist = plistlib.loads(path.read_bytes())
        mode = plist["ProgramArguments"][-1]
        minutes = ", ".join(f":{m['Minute']:02d}" for m in plist["StartCalendarInterval"])
        loaded = subprocess.run(["launchctl", "print", f"{domain()}/{label(fly)}"], capture_output=True).returncode == 0
        print(f"{label(fly)}: {'loaded' if loaded else 'NOT loaded'}, mode {mode}, runs at {minutes}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["install", "remove", "status"])
    ap.add_argument("--mode", choices=["record", "go"], default="record")
    ap.add_argument("--fly", default=None, help="only this fly; default is all of them")
    args = ap.parse_args()
    {"install": lambda: install(args.mode, args.fly), "remove": remove, "status": status}[args.command]()
