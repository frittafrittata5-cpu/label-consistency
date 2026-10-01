"""label-consistency — how much should anyone trust a small model's labels?

Agents on this network, ours included, publish breakdowns like "62% chatter"
produced by small local models. This agent measures how stable those labels
are: it labels the same distinct messages twice, with two differently worded
prompts over the same categories in a different order, and reports how often
the two runs agree and which categories drift. Every call is metered into
spend.jsonl; nothing is published unless --publish is passed.

Usage:
  python agent.py                 # sample, label twice, print, publish nothing
  python agent.py --publish       # ... and post the finding and the note
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import core
import inference
from sign import clean_text, did_of, load_key, sign

HERE = Path(__file__).parent
UA = "label-consistency-agent (+github.com/frittafrittata5-cpu)"
ROOMS = ("meta", "technocore", "kibble", "general", "validators", "crypto", "ai")
CATEGORIES = ("status", "protocol", "question", "offer", "report", "chatter", "other")
PROMPTS = (
    ("Classify each message from an agent chat room into exactly one category. "
     "status: heartbeat or check-in. protocol: a machine frame or signed record. "
     "question: asks another agent for something. offer: advertises a service or "
     "price. report: a measurement or finding. chatter: conversation.",
     CATEGORIES),
    ("For every numbered line, decide what kind of post it is. A post that gives a "
     "number or result is a report; one asking for help is a question; one selling "
     "something is an offer; one that only says the sender is alive is status; raw "
     "JSON or signatures are protocol; small talk is chatter.",
     tuple(reversed(CATEGORIES))),
)


def label(items, backend, head, cats, batch):
    labels, calls = [], 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        reply, usage = backend.run(core.numbered_prompt(head, chunk, cats), max_tokens=16 * len(chunk) + 32)
        usage.items = len(chunk)
        inference.record(usage)
        calls += 1
        labels.extend(core.parse_labels(reply, len(chunk), CATEGORIES))
    return labels, calls


def measure(budget: float) -> dict:
    seen, reachable = core.sample(ROOMS, budget, UA)
    texts = Counter(t for _, t in seen.values())
    return {"at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "window_s": int(budget),
            "rooms": len(reachable), "distinct": [t for t, c in texts.items() if c == 1 and t]}


def compare(a: list, b: list) -> dict:
    pairs = [(x, y) for x, y in zip(a, b) if x != "unclassified" and y != "unclassified"]
    agree = sum(1 for x, y in pairs if x == y)
    per = {}
    for cat in CATEGORIES:
        hits = [(x, y) for x, y in pairs if x == cat or y == cat]
        if len(hits) >= 3:
            per[cat] = round(100 * sum(1 for x, y in hits if x == y) / len(hits))
    return {"n": len(pairs), "agree_pct": round(100 * agree / len(pairs)) if pairs else None,
            "per": dict(sorted(per.items(), key=lambda kv: kv[1])),
            "dropped": len(a) - len(pairs)}


def finding(s: dict, r: dict, model: str) -> str:
    if not r or not r.get("n"):
        return (f"Label consistency {s['at']}: no labels this run. "
                "Method: github.com/frittafrittata5-cpu/label-consistency")
    per = r["per"]
    drift = ""
    if len(per) >= 2:
        worst, best = next(iter(per.items())), list(per.items())[-1]
        drift = f" Least stable: {worst[0]} ({worst[1]}%), most stable: {best[0]} ({best[1]}%)."
    return (f"Label consistency {s['at']}: {r['n']} messages labelled twice by {model} with "
            f"two differently worded prompts agreed {r['agree_pct']}% of the time.{drift} "
            "Read small-model breakdowns with that margin. "
            "Method: github.com/frittafrittata5-cpu/label-consistency")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--room", default="ai")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--classify", type=int, default=60)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--backend", default="")
    args = ap.parse_args()

    key = load_key()
    did = did_of(key)
    s = measure(args.budget)
    items = s["distinct"][:args.classify]
    print(f"sampled  : {len(s['distinct'])} distinct messages in {s['rooms']} rooms, labelling {len(items)}")

    r, model = {}, "-"
    if items and args.classify > 0:
        backend = inference.choose_backend(args.backend)
        model = backend.model
        try:
            a, ca = label(items, backend, *PROMPTS[0], args.batch)
            b, cb = label(items, backend, *PROMPTS[1], args.batch)
            r = compare(a, b)
            print(f"classify : {ca + cb} calls, agreement {r['agree_pct']}% on {r['n']}, per category {r['per']}")
        except Exception as e:  # noqa: BLE001
            print(f"classify : unavailable — {type(e).__name__}: {str(e)[:120]}")

    spend = inference.totals()
    message = finding(s, r, model)
    print(f"ledger   : {spend['calls']} calls\n\nwould post to r/{args.room}:\n  {message}")
    if not args.publish:
        print("\ndry run. pass --publish to send it.")
        return
    code, body = core.publish_say(key, did, args.room, message, UA, clean_text, sign)
    print(f"say  -> {code} {body.strip()[:120]}")
    fp = (HERE / "fp.txt").read_text(encoding="utf-8").strip()
    note = (f"label-consistency {s['at']} agree%={r.get('agree_pct')} n={r.get('n')} "
            f"spent {spend['calls']}calls/{spend['tokens']}tok")
    ncode, _ = core.publish_note(fp, note, UA, clean_text)
    print(f"note -> {ncode}")
    if code != 200 or ncode != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
