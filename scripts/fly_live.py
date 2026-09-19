"""Run one fly against ClawStreet: fetch the board, decide, post.

    python scripts/fly_live.py --fly 001            # dry run: prints the plan and the thought
    python scripts/fly_live.py --fly 001 --go       # places the order and posts the thought
    python scripts/fly_live.py --fly 001 --register # creates the bot, stores its key in the keychain

Each fly is a row in data/flies/flies.json with its universe, cadence and the
keychain item that holds its API key. Market data comes from ClawStreet's
history endpoint; the decision is fly_agent.run_session; orders and thoughts
go to the v1 API. The thought is posted within five seconds of the order so
the feed pairs them.
"""

import argparse
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fly_agent import BOARD_SIZE, MAX_POSITIONS, compose_board, run_session, size
from fly_memory import FLIES

BASE = "https://www.clawstreet.io/api"
MANIFEST = FLIES / "flies.json"
DATA_KEYCHAIN = "clawstreet-turtle-api-key"   # read-only market data until the fly has a key
TIMEOUT = 20


def keychain(item: str) -> str | None:
    out = subprocess.run(["security", "find-generic-password", "-s", item, "-w"],
                         capture_output=True, text=True)
    return out.stdout.strip() or None


def keychain_add(item: str, secret: str) -> None:
    subprocess.run(["security", "add-generic-password", "-s", item, "-a", "fly", "-w", secret, "-U"],
                   check=True, capture_output=True)


def api(key: str | None, method: str, path: str, json_body: dict | None = None,
        headers: dict | None = None) -> dict:
    """One call to ClawStreet. Raises with the status and body on any error."""
    head = {"Content-Type": "application/json", **(headers or {})}
    if key:
        head["Authorization"] = f"Bearer {key}"
    data = json.dumps(json_body).encode() if json_body is not None else None
    try:
        with urlopen(Request(f"{BASE}{path}", data=data, headers=head, method=method), timeout=TIMEOUT) as r:
            return json.loads(r.read())
    except HTTPError as e:
        raise RuntimeError(f"{method} {path} -> {e.code}: {e.read()[:400]!r}") from e


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, indent=1))


# ---- market data ---------------------------------------------------------

def universe(key: str, kind: str) -> list[str]:
    symbols = [s["symbol"] if isinstance(s, dict) else s for s in api(key, "GET", "/data/symbols")["symbols"]]
    crypto = [s for s in symbols if s.startswith("X:")]
    return crypto if kind == "crypto" else [s for s in symbols if s not in crypto]


def history(key: str, symbols: list[str], periods: int = 20) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for i in range(0, len(symbols), 20):
        chunk = symbols[i:i + 20]
        out.update(api(key, "GET", f"/data/history?symbols={quote(','.join(chunk))}&periods={periods}"))
    return {s: out[s] for s in symbols if s in out and out[s].get("derived")}


# ---- what closed since last time -------------------------------------------

def closed_positions(key: str, bot_id: str, fly_positions: list[dict]) -> dict[str, bool]:
    """Positions the fly holds on paper that ClawStreet no longer holds.

    The outcome is the realised result from fills since the position opened:
    what the sells brought in, less what the buys cost, less commission.
    """
    live = {p["symbol"] for p in api(key, "GET", f"/v1/me/agents/{bot_id}/positions")["positions"]}
    gone = [p for p in fly_positions if p["symbol"] not in live]
    if not gone:
        return {}
    fills = api(key, "GET", f"/v1/me/agents/{bot_id}/fills?limit=500")["data"]
    result = {}
    for p in gone:
        since = p["opened"]
        pnl = 0.0
        for f in fills:
            if f["symbol"] != p["symbol"] or f["created_at"][:10] < since:
                continue
            value = f["qty"] * f["price"]
            pnl += value if f["side"] in ("sell", "cover") else -value
            pnl -= f.get("commission") or 0.0
        result[p["symbol"]] = pnl > 0
    return result


# ---- words -------------------------------------------------------------------

def thought(result: dict, board_size: int, qty: float | None, price: float | None, session: int) -> str:
    """What the fly posts. Third person, the numbers as they are, under 500 characters."""
    lines = []
    for sym, won in result["settled"]:
        cells = result["settled_cells"].get(sym)
        if cells is None:
            continue
        lines.append(f"{sym} closed {'up' if won else 'down'}: dopamine to the {cells} cells that chose it, "
                     f"{'reward' if won else 'punishment'} side.")
    for sym in result["sold"]:
        lines.append(f"Selling {sym}: liked it less than at purchase, two sessions running.")
    order = result["order"]
    ranking = result["ranking"]
    if order:
        top, second = ranking[0], ranking[1] if len(ranking) > 1 else None
        head = f"{board_size} on the table. {order['symbol']} smelled best at {order['verdict']:.3f}"
        if second:
            head += f", {second[0]} next at {second[1]:.3f}"
        head += f": {result['pick_cells']} Kenyon cells, {result['channels']} channels."
        spend = qty * price if qty and price else order["dollars"]
        if order["margin"] < 0.36:
            head += f" Too close to tell apart, so half size: ${spend:,.0f}"
        else:
            head += f" Clear of the rest by {order['margin']:.2f}: ${spend:,.0f}"
        head += f", {qty:g} at ${price:,.2f}." if qty and price else "."
        lines.append(head)
    else:
        why = "holds the maximum already" if len(result["held"]) >= MAX_POSITIONS else "nothing on the table it could buy"
        lines.append(f"{board_size} on the table, nothing bought: {why}.")
    drift = result["drift"]
    lines.append(f"Session {session}. Synapses {drift:.3f} from the connectome" +
                 (", nothing learned yet." if drift == 0 else "."))
    text = " ".join(lines)
    return text if len(text) <= 500 else text[:497] + "..."


# ---- the run --------------------------------------------------------------------

def register(fly_id: str, m: dict) -> None:
    row = m[fly_id]
    body = {
        "name": row["name"],
        "ticker": row["ticker"],
        "strategy": ("A fruit fly connectome (MaleCNS v1.0, 164,587 neurons) smells stock setups. "
                     "Six indicators become 51 receptor channels; the olfactory circuit runs for 50 ms; "
                     "the Kenyon cells that fire decide. Buys what smells best, sells what it likes less "
                     "two sessions running, learns at the synapse with a dopamine rule, forgets 2% a session."),
        "bio": "A fly, sniffing for alpha. Born a momentum trader; nobody taught it anything.",
        # Both become public pages (/models/<slug>, /frameworks/<slug>), so they
        # have to read clearly to someone who has never heard of a connectome.
        "model": "Fruit Fly Brain",
        "framework": "Python + Brian2",
    }
    r = api(None, "POST", "/bots/register", json_body=body)
    if not r.get("ok", r.get("success")):
        raise RuntimeError(f"register failed: {r}")
    keychain_add(row["keychain"], r["api_key"])
    row["bot_id"] = r["bot_id"]; row["claim_url"] = r["claim_url"]
    row["started"] = datetime.now(timezone.utc).isoformat(timespec="minutes")
    save_manifest(m)
    print(f"registered {row['name']} as {r['bot_id']}")
    print(f"claim it here: {r['claim_url']}  (code {r.get('verification_code')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default="001")
    ap.add_argument("--go", action="store_true", help="place the order and post the thought")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="board seed; default is the minute")
    args = ap.parse_args()

    m = load_manifest()
    if args.fly not in m:
        raise SystemExit(f"no fly {args.fly} in {MANIFEST}")
    row = m[args.fly]
    if args.register:
        register(args.fly, m); return

    key = keychain(row["keychain"]) or keychain(DATA_KEYCHAIN)
    if not key:
        raise SystemExit("no API key in the keychain")
    bot_id = row.get("bot_id")
    if args.go and not bot_id:
        raise SystemExit("register the fly first: --register, then claim it")

    when = datetime.now(timezone.utc)
    home = FLIES / args.fly
    positions = json.loads((home / "positions.json").read_text()) if (home / "positions.json").exists() else {"open": [], "pending": []}
    held = {p["symbol"] for p in positions["open"]}

    account = 100_000.0
    closed: dict[str, bool] = {}
    if bot_id and keychain(row["keychain"]):
        portfolio = api(key, "GET", f"/v1/me/agents/{bot_id}/portfolio")
        account = float(portfolio["equity"])
        closed = closed_positions(key, bot_id, positions["open"] + positions["pending"])

    seed = args.seed if args.seed is not None else int(when.strftime("%Y%m%d%H%M"))
    symbols = compose_board(held, universe(key, row["universe"]), seed, BOARD_SIZE)
    board = history(key, symbols)
    if len(board) < 2:
        raise SystemExit(f"history came back thin: {list(board)}")

    result = run_session(board, closed, dry_run=not args.go, fly_id=args.fly, when=when, account=account)
    order = result["order"]
    qty = price = None
    if order:
        price = float(board[order["symbol"]]["current_price"])
        raw = order["dollars"] / price
        qty = round(raw, 5) if order["symbol"].startswith("X:") else float(int(raw))
        if qty <= 0:
            order = None; result["order"] = None
    session = result["session"] + (0 if args.go else 1)
    text = thought(result, len(board), qty, price, session)

    print(f"fly {args.fly} · session {session} · {when.strftime('%Y-%m-%d %H:%M')} UTC · equity ${account:,.0f}")
    print("board:   " + ", ".join(board))
    print("settled: " + (", ".join(f"{s} {'up' if w else 'down'}" for s, w in result["settled"]) or "nothing"))
    print("ranking: " + ", ".join(f"{s} {v:.3f}" for s, v in result["ranking"]))
    print("sold:    " + (", ".join(result["sold"]) or "nothing"))
    if order:
        print(f"order:   buy {qty:g} {order['symbol']} at ${price:,.2f} = ${qty * price:,.2f} of a ${order['dollars']:,.2f} budget (verdict {order['verdict']:.3f}, margin {order['margin']:.3f})")
    else:
        print("order:   none")
    print(f"thought ({len(text)} chars):\n  {text}")
    if not args.go:
        print("\ndry run. Nothing sent, nothing saved.")
        return

    for sym in result["sold"]:
        held_qty = next((p["qty"] for p in positions["open"] if p["symbol"] == sym), 0)
        if held_qty > 0:
            api(key, "POST", f"/v1/me/agents/{bot_id}/orders", json_body={"symbol": sym, "side": "sell", "qty": held_qty, "type": "market",
                "reasoning": f"Liked {sym} less than at purchase, two sessions running."},
                headers={"Idempotency-Key": str(uuid.uuid4())})
            time.sleep(1)
    if order:
        placed = api(key, "POST", f"/v1/me/agents/{bot_id}/orders", json_body={
            "symbol": order["symbol"], "side": "buy", "qty": qty, "type": "market",
            "reasoning": text + "\n\nWhat it smelled of:\n" + next(e["smell"] for e in reversed(json.loads(Path(result["replay"]).read_text())["events"]) if e["type"] == "buy")},
            headers={"Idempotency-Key": str(uuid.uuid4())})
        print("order placed:", placed.get("data", placed).get("id", placed))
        # Write the real quantity and price back onto the paper position.
        pos = json.loads((home / "positions.json").read_text())
        for p in pos["open"]:
            if p["symbol"] == order["symbol"] and p["qty"] == 0.0:
                p["qty"] = qty; p["price"] = price
        (home / "positions.json").write_text(json.dumps(pos, indent=1))
        time.sleep(1)
    api(key, "POST", f"/v1/me/agents/{bot_id}/thoughts", json_body={"body": text})
    print("thought posted")


if __name__ == "__main__":
    main()
