"""Install or remove the launchd jobs that run the flies on this Mac.

    python scripts/fly_schedule.py install --mode record   # rehearsals: real data, real brain, no orders
    python scripts/fly_schedule.py install --mode go       # live: places orders and posts thoughts
    python scripts/fly_schedule.py remove
    python scripts/fly_schedule.py status

One job per fly in data/flies/flies.json. launchd calls the runner on the hour
and the half hour; the runner's --if-due check decides whether a session is
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


def install(mode: str) -> None:
    AGENTS.mkdir(parents=True, exist_ok=True)
    for fly in json.loads((FLIES / "flies.json").read_text()):
        (FLIES / fly).mkdir(parents=True, exist_ok=True)
        log = str(FLIES / fly / "run.log")
        plist = {
            "Label": label(fly),
            "ProgramArguments": [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/fly_live.py"),
                                 "--fly", fly, "--if-due", f"--{mode}"],
            "WorkingDirectory": str(ROOT),
            "StartCalendarInterval": [{"Minute": 0}, {"Minute": 30}],
            "EnvironmentVariables": {"PATH": PATH},
            "StandardOutPath": log,
            "StandardErrorPath": log,
            "ProcessType": "Background",
        }
        path = AGENTS / f"{label(fly)}.plist"
        unload(fly)
        path.write_bytes(plistlib.dumps(plist))
        subprocess.run(["launchctl", "bootstrap", domain(), str(path)], check=True)
        print(f"{label(fly)}: installed, mode {mode}, log {log}")


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
        mode = plistlib.loads(path.read_bytes())["ProgramArguments"][-1]
        loaded = subprocess.run(["launchctl", "print", f"{domain()}/{label(fly)}"], capture_output=True).returncode == 0
        print(f"{label(fly)}: {'loaded' if loaded else 'NOT loaded'}, mode {mode}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["install", "remove", "status"])
    ap.add_argument("--mode", choices=["record", "go"], default="record")
    args = ap.parse_args()
    {"install": lambda: install(args.mode), "remove": remove, "status": status}[args.command]()
