"""Synthesis of a minimal-cost enforcement policy.

Problem
-------
Choose gates (edge, mode) so that

1. no witness survives (every integrity and confidentiality witness in G_T
   is broken by at least one gate), and
2. every task requirement still holds (no gate blocks a required edge under
   the label the task needs), and
3. the total utility cost  sum_e c(e, mode_e)  is minimal.

Exact method: integer linear programming with row generation
------------------------------------------------------------
Enumerating all witnesses up front is impossible in general (there can be
exponentially many paths). Instead we use *row generation* (also called
lazy constraint generation), a standard technique in combinatorial
optimisation:

    constraints = {}
    repeat
        policy  = optimal solution of the ILP with the current constraints
        result  = verify(policy)                     # static verifier
        if result is safe: stop                      # policy is optimal
        add one constraint per witness found:  sum of its blocking options >= 1

Why the result is optimal: every added constraint is *necessary* (any
secure policy must break that witness), so the ILP optimum is a lower bound
on the true optimum at every iteration. When the verifier finds no witness,
the current solution is secure and its cost equals the lower bound.
Termination: each iteration adds a witness not yet covered, and there are
finitely many derivations in the finite graph.

Variables:   y[e, m] in {0, 1}   (gate edge e in mode m)
Objective:   minimise  sum c(e, m) * y[e, m]
Constraints: sum_m y[e, m] <= 1                     (one mode per edge)
             sum_{(e,m) in options(w)} y[e, m] >= 1 (for each witness w)
Forbidden pairs (fixed edges, or modes that break a task) get no variable.

If a witness has no allowed option at all, it cannot be broken without
breaking a task: this is reported as an explicit *security-utility
conflict* instead of silently weakening security or utility.

Polynomial special case: minimum s-t cut
-----------------------------------------
If only integrity is considered and no flow crosses sessions, the problem
is a classic minimum cut: a super-source feeds every untrusted tool output,
every integrity sink feeds a super-sink, and an edge's capacity is the cost
of an ``I`` gate on it (infinite if the edge cannot be gated). Max-flow /
min-cut solves this exactly in polynomial time. :func:`synthesize_mincut`
implements it; the test suite checks that it gives the same cost as the ILP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Sequence

import networkx as nx
import pulp

from .graph import CompiledGraph
from .labels import UNTRUSTED, Mode, blocks, breaks_task, forbidden_modes_for_requirement
from .policy import Policy
from .verify import ALL_PROPERTIES, Property, Witness, verify


@dataclass
class Conflict:
    """A witness that cannot be broken without breaking a task."""

    witness: Witness
    reasons: list[str]  # one explanation per step


@dataclass
class SynthesisResult:
    status: Literal["secure", "conflict"]
    policy: Policy
    cost: float
    iterations: int
    witnesses: list[Witness] = field(default_factory=list)  # constraints used
    conflicts: list[Conflict] = field(default_factory=list)
    solve_seconds: float = 0.0
    method: str = "ilp"

    def summary(self) -> dict:
        return {
            "status": self.status,
            "method": self.method,
            "cost": round(self.cost, 6),
            "gates": len(self.policy.gates),
            "iterations": self.iterations,
            "constraints": len(self.witnesses),
            "conflicts": len(self.conflicts),
            "solve_seconds": round(self.solve_seconds, 4),
        }


# --- Helpers ----------------------------------------------------------------


def forbidden_pairs(cg: CompiledGraph) -> dict[str, set[Mode]]:
    """Modes that may not be placed on each edge (fixed edges and task needs).

    Human confirmation (H) is only possible on ``call`` edges (an action a
    human can approve) and only if ``controls.allow_human`` is set.
    """
    m = cg.manifest
    forbidden: dict[str, set[Mode]] = {}
    for edge, tpl in cg.templates.items():
        if tpl.kind != "call" or not m.controls.allow_human:
            forbidden[edge] = {Mode.HUMAN}
    for edge in m.controls.fixed:
        forbidden[edge] = set(Mode)  # replaces the H-only entry
    for task in m.tasks:
        for req in task.requires:
            forbidden.setdefault(req.edge, set()).update(
                forbidden_modes_for_requirement(req.label))
    return forbidden


def _explain(cg: CompiledGraph, witness: Witness) -> list[str]:
    """Explain, step by step, why a witness cannot be broken."""
    m = cg.manifest
    needs: dict[str, list[str]] = {}
    for task in m.tasks:
        for req in task.requires:
            needs.setdefault(req.edge, []).append(f"{task.id} needs it under {sorted(req.label)}")
    reasons = []
    for tpl in witness.templates():
        if tpl in m.controls.fixed:
            reasons.append(f"{tpl}: listed in controls.fixed (no enforcement point)")
        else:
            reasons.append(f"{tpl}: " + "; ".join(needs.get(tpl, ["no allowed mode"])))
    return reasons


def _tie_break(edge: str, mode: Mode, rank: dict[str, int]) -> float:
    # Tiny deterministic perturbation so that, among equal-cost optima, the
    # solver always returns the same policy (reproducibility of results).
    return 1e-6 * (rank[edge] * 4 + "RICH".index(mode.value) + 1) / (4 * len(rank) + 4)


def _solve_ilp(cg: CompiledGraph, constraints: list[set[tuple[str, Mode]]],
               warm: Optional[dict[str, Mode]], time_limit: Optional[float],
               profiles: Sequence[tuple[float, frozenset]] = (),
               workflow_weight: float = 0.0) -> dict[str, Mode]:
    """Solve the hitting-set ILP for the current set of constraints.

    ``profiles`` (optional) are the usage profiles of legitimate workflows,
    each with a weight. For each workflow a binary variable z_w is forced to
    1 as soon as one chosen gate blocks one of its (edge, label) pairs, and
    ``workflow_weight * weight * z_w`` is added to the objective. The
    solver then trades gate cost against broken workflows.
    """
    m = cg.manifest
    pairs = sorted({p for opts in constraints for p in opts}, key=lambda p: (p[0], p[1].value))
    rank = {e: i for i, e in enumerate(sorted(cg.templates))}

    prob = pulp.LpProblem("amps_policy_synthesis", pulp.LpMinimize)
    y = {p: pulp.LpVariable(f"y_{i}", cat="Binary") for i, p in enumerate(pairs)}

    prob += pulp.lpSum((m.edge_cost(e, md) + _tie_break(e, md, rank)) * var
                       for (e, md), var in y.items())
    # At most one mode per edge.
    by_edge: dict[str, list] = {}
    for (e, _md), var in y.items():
        by_edge.setdefault(e, []).append(var)
    for e, vars_ in by_edge.items():
        if len(vars_) > 1:
            prob += pulp.lpSum(vars_) <= 1, f"one_mode_{rank[e]}"
    # Every known witness must be broken.
    for i, opts in enumerate(constraints):
        prob += pulp.lpSum(y[p] for p in opts) >= 1, f"witness_{i}"

    # Workflow-weighted objective: z_w = 1 if a chosen gate breaks workflow w.
    if workflow_weight > 0:
        for j, (weight, profile) in enumerate(profiles):
            breakers = [var for (e, md), var in y.items()
                        if any(edge == e and breaks_task(md, lbl) for edge, lbl in profile)]
            if not breakers:
                continue  # no candidate gate can break this workflow
            z = pulp.LpVariable(f"z_{j}", cat="Binary")
            for k, var in enumerate(breakers):
                prob += z >= var, f"breaks_{j}_{k}"
            prob.objective += workflow_weight * weight * z

    use_warm = bool(warm)
    if use_warm:
        for (e, md), var in y.items():
            var.setInitialValue(1 if warm.get(e) is md else 0)
    solver = pulp.PULP_CBC_CMD(msg=False, warmStart=use_warm, timeLimit=time_limit)
    prob.solve(solver)
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"ILP solver status: {pulp.LpStatus[prob.status]}")
    return {e: md for (e, md), var in y.items() if var.value() is not None and var.value() > 0.5}


# --- Exact synthesis (ILP + row generation) ---------------------------------


def synthesize_ilp(cg: CompiledGraph,
                   properties: Iterable[Property] = ALL_PROPERTIES,
                   warm_start: Optional[Policy] = None,
                   initial_witnesses: Iterable[Witness] = (),
                   max_iterations: int = 1000,
                   time_limit: Optional[float] = None,
                   workflows: Sequence[tuple[float, frozenset]] = (),
                   workflow_weight: Optional[float] = None) -> SynthesisResult:
    """Compute a minimum-cost secure policy (see module docstring).

    Workflow-weighted synthesis
    ---------------------------
    ``workflows`` is a list of ``(weight, usage_profile)`` pairs, typically
    built with :func:`amps.workflows.usage_profile` from the task workflows
    and the mixed workflows. The objective becomes

        minimise  sum_e c(e, m) y[e, m]  +  lambda * sum_w weight_w * z_w

    where z_w = 1 if workflow w is broken by the policy. With
    ``workflow_weight=None`` (default) lambda is larger than the cost of any
    possible policy, so the solver first minimises the (weighted) number of
    broken workflows and only then the gate cost (lexicographic order).
    Security constraints are unchanged: the result is still secure, and
    still optimal for the new objective (the row-generation argument does
    not depend on the objective).
    """
    t0 = time.perf_counter()
    properties = tuple(properties)
    m = cg.manifest
    forbidden = forbidden_pairs(cg)
    fixed = set(m.controls.fixed)

    def allowed_options(w: Witness) -> set[tuple[str, Mode]]:
        return {(e, md) for (e, md) in w.blocking_options(fixed)
                if md not in forbidden.get(e, set())}

    profiles = list(workflows)
    if profiles and workflow_weight is None:
        # Lexicographic: larger than the cost of gating every edge with R.
        workflow_weight = 1.0 + sum(m.edge_cost(e, Mode.REMOVE) for e in cg.templates)
    lam = workflow_weight or 0.0

    def solve(cons, warm):
        return _solve_ilp(cg, cons, warm, time_limit, profiles, lam)

    constraints: list[set[tuple[str, Mode]]] = []
    used: list[Witness] = []
    conflicts: list[Conflict] = []
    unavoidable: set[tuple[str, Property]] = set()
    known: set[tuple] = set()

    def register(w: Witness) -> bool:
        """Add the constraint of ``w``; return False if it is a conflict."""
        if w.signature() in known:
            return True
        known.add(w.signature())
        opts = allowed_options(w)
        if not opts:
            conflicts.append(Conflict(w, _explain(cg, w)))
            unavoidable.add((w.sink_tool, w.property))
            return False
        constraints.append(opts)
        used.append(w)
        return True

    # Warm start: keep the old gates that are still valid in this manifest.
    gates: dict[str, Mode] = {}
    if warm_start is not None:
        gates = {e: md for e, md in warm_start.gates.items()
                 if e in cg.templates and md not in forbidden.get(e, set())}
    for w in initial_witnesses:
        if w.property in properties and all(s.template in cg.templates for s in w.steps):
            register(w)
    if constraints:
        gates = solve(constraints, gates)

    iterations = 0
    while True:
        iterations += 1
        if iterations > max_iterations:
            raise RuntimeError("row generation did not converge (raise max_iterations)")
        result = verify(cg, gates, properties, ignore=unavoidable)
        if result.safe:
            break
        before = len(known)
        for w in result.witnesses:
            register(w)  # adds a constraint, or records a conflict
        if len(known) == before:  # pragma: no cover - defensive
            # The verifier returned only witnesses we already had: this would
            # mean the ILP ignored a constraint. Fail loudly instead of looping.
            raise RuntimeError("no progress in row generation")
        if constraints:
            gates = solve(constraints, gates)

    policy = Policy.for_manifest(m, gates)
    cost = policy.cost(m)
    status = "conflict" if conflicts else "secure"
    policy.meta.update({"method": "ilp", "status": status, "cost": round(cost, 6),
                        "properties": list(properties), "sessions": cg.sessions,
                        "iterations": iterations, "constraints": len(used),
                        "conflicts": [{"sink": c.witness.sink_tool,
                                       "property": c.witness.property} for c in conflicts]})
    if profiles:
        broken = sum(w for w, prof in profiles
                     if any(breaks_task(gates.get(e), lbl) for e, lbl in prof))
        policy.meta.update({"objective": "workflow-weighted", "workflows": len(profiles),
                            "broken_workflow_weight": broken, "workflow_weight": lam})
    return SynthesisResult(status, policy, cost, iterations, used, conflicts,
                           time.perf_counter() - t0, "ilp")


# --- Polynomial special case (min s-t cut) ----------------------------------


def synthesize_mincut(cg: CompiledGraph) -> SynthesisResult:
    """Integrity-only synthesis by minimum s-t cut.

    Valid when no flow crosses sessions (no persistent memory, or T = 1): the
    session copies are then identical and independent, so we cut session 1
    and apply the same gates to every session. Raises ``ValueError``
    otherwise, because shared gates across linked copies are no longer a
    plain cut problem (use :func:`synthesize_ilp`).
    """
    t0 = time.perf_counter()
    if cg.has_cross_session_edges():
        raise ValueError("min-cut requires no cross-session flow; use synthesize_ilp")
    m = cg.manifest
    forbidden = forbidden_pairs(cg)
    g = cg.graph

    flow = nx.DiGraph()
    SRC, SNK = "__source__", "__sink__"
    for u, v, data in g.edges(data=True):
        sessions = {s for s in (g.nodes[u].get("session"), g.nodes[v].get("session")) if s}
        if sessions and sessions != {1}:
            continue  # keep only the session-1 copy
        tpl = data["template"]
        if Mode.INTEGRITY in forbidden.get(tpl, set()):
            flow.add_edge(u, v)  # no 'capacity' attribute = infinite capacity
        else:
            flow.add_edge(u, v, capacity=m.edge_cost(tpl, Mode.INTEGRITY), template=tpl)
    for n, data in g.nodes(data=True):
        if data["kind"] == "out" and UNTRUSTED in data["label"]:
            flow.add_edge(SRC, n)
        if data["kind"] == "in" and data["integrity_sink"]:
            flow.add_edge(n, SNK)

    gates: dict[str, Mode] = {}
    status = "secure"
    conflicts: list[Conflict] = []
    if SRC in flow and SNK in flow and nx.has_path(flow, SRC, SNK):
        try:
            _value, (reach, _non_reach) = nx.minimum_cut(flow, SRC, SNK)
            for u, v, data in flow.edges(data=True):
                if u in reach and v not in reach and "template" in data:
                    gates[data["template"]] = Mode.INTEGRITY
        except nx.NetworkXUnbounded:
            # An infinite-capacity path exists: some integrity violation
            # cannot be prevented without breaking a task. Report it.
            status = "conflict"
            for w in verify(cg, {}, ("integrity",)).witnesses:
                conflicts.append(Conflict(w, _explain(cg, w)))

    policy = Policy.for_manifest(m, gates)
    cost = policy.cost(m)
    policy.meta.update({"method": "mincut", "status": status, "cost": round(cost, 6),
                        "properties": ["integrity"], "sessions": cg.sessions})
    if status == "secure":
        # Soundness check: the cut must leave no integrity witness.
        assert verify(cg, gates, ("integrity",)).safe, "min-cut produced an unsafe policy"
    return SynthesisResult(status, policy, cost, 1, [], conflicts,
                           time.perf_counter() - t0, "mincut")


def synthesize(cg: CompiledGraph, method: Literal["auto", "ilp", "mincut"] = "auto",
               properties: Iterable[Property] = ALL_PROPERTIES, **kwargs) -> SynthesisResult:
    """Front door. ``auto`` uses min-cut when it is exact, otherwise the ILP."""
    properties = tuple(properties)
    if method == "mincut" or (method == "auto" and properties == ("integrity",)
                              and not cg.has_cross_session_edges()
                              and not cg.manifest.controls.allow_human):
        return synthesize_mincut(cg)
    return synthesize_ilp(cg, properties, **kwargs)
