"""Metrics and confidence intervals.

Proportions (success rates) are reported with the Wilson score interval
at 95 %. It is preferred to the usual "p ± 1.96·sqrt(p(1-p)/n)" interval
because it stays inside [0, 1] and remains reliable for small samples and
for proportions close to 0 or 1, which is exactly our situation (few
workflows, success rates near 100 %). Reference: Wilson (1927); Brown, Cai
and DasGupta (2001).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable

from .workflows import WorkflowResult


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes out of n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Rate:
    k: int
    n: int

    @property
    def value(self) -> float:
        return self.k / self.n if self.n else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.k, self.n)

    def fmt(self) -> str:
        lo, hi = self.ci
        return f"{100 * self.value:5.1f}% [{100 * lo:5.1f}, {100 * hi:5.1f}] ({self.k}/{self.n})"


def utility_summary(results: Iterable[WorkflowResult]) -> dict:
    """Utility metrics of one configuration.

    ``UA_task``   share of single-task workflows that succeed
    ``UA_mixed``  share of mixed workflows that succeed
    ``UA_safe``   share of *label-safe* workflows that succeed: the utility
                    that a secure policy could keep in principle (100 % is
                    the best possible); its complement is the false-block rate
    ``unsafe_passed`` share of *label-unsafe* workflows that a policy lets
                    through: these are information-flow violations (0 % is
                    required for a secure policy)
    ``block_rate`` share of all actions that were blocked
    ``check_us_per_action`` mean time of one policy check (overhead)
    """
    results = list(results)
    task = [r for r in results if r.kind == "task"]
    mixed = [r for r in results if r.kind == "mixed"]
    actions = sum(r.n_actions for r in results)
    blocked = sum(r.n_blocked for r in results)
    blockers: dict[str, int] = {}
    for r in results:
        if r.first_block:
            blockers[r.first_block] = blockers.get(r.first_block, 0) + 1
    return {
        "UA_task": Rate(sum(r.success for r in task), len(task)),
        "UA_mixed": Rate(sum(r.success for r in mixed), len(mixed)),
        "UA_safe": Rate(sum(r.success for r in results if r.label_safe),
                        sum(r.label_safe for r in results)),
        "unsafe_passed": Rate(sum(r.success for r in results if not r.label_safe),
                              sum(not r.label_safe for r in results)),
        "block_rate": Rate(blocked, actions),
        "filtered_entries": sum(r.n_filtered for r in results),
        "check_us_per_action": (sum(r.check_us for r in results) / actions) if actions else 0.0,
        "median_workflow_us": statistics.median(r.wall_us for r in results) if results else 0.0,
        "top_blockers": dict(sorted(blockers.items(), key=lambda kv: -kv[1])[:5]),
    }
