"""Security labels and gate modes.

The module defines two small vocabularies that the rest of the
framework is built on.

1. Labels
   A *label* is the set of security-relevant origins carried by a piece of
   information. We only track two origins, because the threat model of the
   framework only needs two:

   * ``"U"`` (Untrusted): the information was produced or influenced by a
     source that an attacker may control (a web page, an incoming e-mail,
     a third-party tool output). Untrusted content is the carrier of
     indirect prompt injection.
   * ``"S"`` (Sensitive): the information contains data whose
     confidentiality must be protected (customer records, private e-mails).

   A label is represented as a ``frozenset`` of these letters, e.g.
   ``frozenset({"U", "S"})``. ``frozenset`` is used (rather than ``set``)
   because it is immutable and hashable, so it can be used as a dictionary
   key or stored inside another set.

2. Gate modes
   A *gate* is a check placed on one edge of the attack-path graph (for
   instance "agent `executor` calls tool `send_email`"). A gate has a mode
   that says which flows it stops:

   * ``R`` (Remove): the edge is removed entirely (classic least privilege).
     It blocks every flow, but it also removes the capability for benign
     tasks, so it is the most expensive mode.
   * ``I`` (Integrity / taint gate): the edge is blocked only when the flow
     carries ``U``. Benign, clean sessions keep the capability.
   * ``C`` (Confidentiality / "trifecta" gate): the edge is blocked only
     when the flow carries both ``U`` and ``S``, i.e. when untrusted
     content and sensitive data are mixed in the same context.
   * ``H`` (Human confirmation): when the flow carries ``U``, the
     action is not executed automatically but submitted to a human, who
     approves it (legitimate task) or rejects it (attack). For *security*
     it acts like ``I`` under the stated assumption that the human rejects
     the attacker's action; for *utility* it never breaks a task, it only
     costs a confirmation. Only allowed on ``call`` edges (actions), and
     only if the manifest enables it (``controls.allow_human``).

Two functions give the semantics of a gate, and every module uses them:

* :func:`blocks` (security view): does the gate stop an attacker's flow?
* :func:`breaks_task` (utility view): does the gate stop a legitimate task?

They differ only for ``H``. :func:`blocks` is the single source of truth for
the security semantics. Both the static verifier (``verify.py``) and the runtime engine
(``runtime.py``) call it, which guarantees that what is proved statically
is exactly what is enforced at run time.
"""

from __future__ import annotations

from enum import Enum
from typing import FrozenSet, Iterable, Optional

# --- Labels -----------------------------------------------------------------

UNTRUSTED = "U"
SENSITIVE = "S"

#: Type alias used in type hints throughout the package.
Label = FrozenSet[str]

#: The empty label: information with no security-relevant origin.
EMPTY: Label = frozenset()

#: The three non-empty labels. The verifier never needs to track the empty
#: label, because a flow with no origin can never cause a violation.
NON_EMPTY_LABELS = (
    frozenset({UNTRUSTED}),
    frozenset({SENSITIVE}),
    frozenset({UNTRUSTED, SENSITIVE}),
)


def make_label(items: Iterable[str]) -> Label:
    """Build a label from an iterable such as ``["U", "S"]``.

    Raises ``ValueError`` on unknown letters, so that a typo in a manifest
    (for example ``"X"``) is caught early instead of silently ignored.
    """
    label = frozenset(items)
    unknown = label - {UNTRUSTED, SENSITIVE}
    if unknown:
        raise ValueError(f"Unknown label element(s): {sorted(unknown)}")
    return label


def fmt(label: Label) -> str:
    """Human-readable form of a label, e.g. ``{S,U}`` or ``{}``."""
    return "{" + ",".join(sorted(label)) + "}"


def integrity_violation(label: Label) -> bool:
    """An integrity violation: untrusted content reaches a state-changing sink."""
    return UNTRUSTED in label


def confidentiality_violation(label: Label) -> bool:
    """A confidentiality violation ("lethal trifecta"): untrusted content and
    sensitive data are both present when an external sink is invoked."""
    return UNTRUSTED in label and SENSITIVE in label


# --- Gate modes -------------------------------------------------------------


class Mode(str, Enum):
    """Gate modes. Inheriting from ``str`` makes the values JSON-friendly."""

    REMOVE = "R"
    INTEGRITY = "I"
    TRIFECTA = "C"
    HUMAN = "H"


def blocks(mode: Optional[Mode], label: Label) -> bool:
    """Return ``True`` if a gate in ``mode`` stops a flow carrying ``label``.

    ``mode is None`` means "no gate on this edge": nothing is blocked.
    """
    if mode is None:
        return False
    if mode is Mode.REMOVE:
        return True
    if mode is Mode.INTEGRITY:
        return UNTRUSTED in label
    if mode is Mode.TRIFECTA:
        return UNTRUSTED in label and SENSITIVE in label
    if mode is Mode.HUMAN:
        # Security view: the human reviewer rejects the attacker's action.
        return UNTRUSTED in label
    raise ValueError(f"Unknown mode: {mode!r}")  # defensive, should not happen


def needs_confirmation(mode: Optional[Mode], label: Label) -> bool:
    """``True`` if the flow is submitted to a human instead of executed directly."""
    return mode is Mode.HUMAN and UNTRUSTED in label


def breaks_task(mode: Optional[Mode], label: Label) -> bool:
    """Utility view: would a legitimate task that needs this flow fail?

    Same as :func:`blocks`, except that a human-confirmation gate never
    breaks a legitimate task (the human approves it); it only costs a
    confirmation.
    """
    if mode is Mode.HUMAN:
        return False
    return blocks(mode, label)


def effective_modes(label: Label) -> set[Mode]:
    """All modes that would block a flow carrying ``label``.

    Used by the synthesis step: to break a witness, at least one of its steps
    must be gated in a mode that is effective for the label on that step.
    """
    return {m for m in Mode if blocks(m, label)}


def forbidden_modes_for_requirement(label: Label) -> set[Mode]:
    """Modes that would break a task that needs an edge under ``label``.

    A task requirement says "this edge must keep working even when the flow
    carries ``label``". Any mode that breaks a task carrying ``label`` is
    forbidden, and ``REMOVE`` always breaks it. ``H`` is never forbidden by
    a requirement (it asks a human instead of failing).
    """
    return {Mode.REMOVE} | {m for m in Mode if breaks_task(m, label)}
