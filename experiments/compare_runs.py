"""Compare all AgentDojo banking runs found under runs/ (attacked cases).

For every pipeline folder runs/<run>/<pipeline>/banking/ that contains
important_instructions results, the script prints the attack success rate
(ASR) and the utility under attack with Wilson 95% intervals. It then compares
every pipeline with the reference pipeline (default: local-amps_BH) on the
same (user task, injection task) pairs with an exact McNemar test.

It also summarizes the human decisions logged by the B+H attack run.

Usage (from the project folder):
    python experiments/compare_runs.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = sys.argv[1] if len(sys.argv) > 1 else "local-amps_BH"
ATTACK = sys.argv[2] if len(sys.argv) > 2 else "important_instructions"
CALL_LOG = ROOT / "results" / ("qwen3-4b-instruct_BH_attack_calls.jsonl" if ATTACK == "important_instructions" else f"qwen3-4b-instruct_BH_{ATTACK}_attack_calls.jsonl")
ATTACKER_IBAN = "US133000000121212121212"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, 100 * (c - h)), min(100.0, 100 * (c + h)))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def load_runs() -> dict[str, dict[tuple[str, str], dict]]:
    """pipeline label -> {(user_task, injection_task): {'security', 'utility'}}"""
    runs = {}
    for f in sorted(ROOT.glob(f"runs/*/*/banking/user_task_*/{ATTACK}/injection_task_*.json")):
        pipeline_dir = f.parents[3]
        label = f"{pipeline_dir.parent.name}/{pipeline_dir.name}"
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (f.parents[1].name, f.stem)
        runs.setdefault(label, {})[key] = {
            "security": bool(d.get("security")),
            "utility": bool(d.get("utility")),
        }
    return runs


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No attacked results found under runs/.")
        return

    print(f"{'pipeline':60s} {'n':>4s} {'ASR':>18s} {'utility':>18s}")
    for label, res in runs.items():
        n = len(res)
        a = sum(r["security"] for r in res.values())
        u = sum(r["utility"] for r in res.values())
        la, ha = wilson(a, n)
        lu, hu = wilson(u, n)
        print(f"{label:60s} {n:4d} {a:3d} [{la:4.1f};{ha:5.1f}] {u:3d} [{lu:4.1f};{hu:5.1f}]")

    refs = [l for l in runs if l.endswith("/" + REFERENCE)]
    if not refs:
        print(f"\nReference pipeline '{REFERENCE}' not found.")
        return
    ref = runs[refs[0]]
    print(f"\nPaired comparison with {refs[0]} (exact McNemar, same cases only)")
    for label, res in runs.items():
        if label == refs[0]:
            continue
        common = sorted(set(ref) & set(res))
        for metric in ("security", "utility"):
            # b: other true, reference false ; c: other false, reference true
            b = sum(res[k][metric] and not ref[k][metric] for k in common)
            c = sum(ref[k][metric] and not res[k][metric] for k in common)
            print(f"  {label:58s} {metric:8s} n={len(common):3d} "
                  f"other-only={b:3d} ref-only={c:3d} p={mcnemar_exact(b, c):.3g}")

    if CALL_LOG.exists():
        print(f"\nHuman decisions in {CALL_LOG.relative_to(ROOT)}")
        rows = [json.loads(line) for line in CALL_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
        reasons = Counter((r.get("reason"), r.get("tool")) for r in rows
                          if r.get("reason") in ("confirmed", "rejected_by_human"))
        for (reason, tool), k in sorted(reasons.items()):
            print(f"  {reason:18s} {tool:30s} {k:4d}")
        to_attacker = Counter(r.get("reason") for r in rows
                              if ATTACKER_IBAN.lower() in json.dumps(r.get("args"), default=str).lower())
        print(f"  calls naming the attacker IBAN, by decision: {dict(to_attacker)}")
        null_task = sum(1 for r in rows if r.get("task") is None
                        and r.get("reason") in ("confirmed", "rejected_by_human"))
        print(f"  confirmation requests with unknown task: {null_task}")


if __name__ == "__main__":
    main()