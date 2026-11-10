from amps.graph import compile_manifest
from amps.labels import Mode
from amps.manifest import manifest_from_dict
from amps.verify import verify

from conftest import tiny


def _cg(d):
    return compile_manifest(manifest_from_dict(d))


def test_tiny_has_integrity_and_confidentiality_paths():
    res = verify(_cg(tiny()))
    kinds = {(w.property, w.sink_tool) for w in res.witnesses}
    assert ("integrity", "email") in kinds
    assert ("confidentiality", "email") in kinds
    # The web tool is itself an external sink: the query can leak PII.
    assert ("confidentiality", "web") in kinds
    assert not res.safe


def test_integrity_gate_blocks_untrusted_but_trifecta_gate_does_not():
    cg = _cg(tiny())
    res_i = verify(cg, {"call:actor->email": Mode.INTEGRITY}, ("integrity",))
    assert res_i.safe
    res_c = verify(cg, {"call:actor->email": Mode.TRIFECTA}, ("integrity",))
    assert not res_c.safe


def test_context_accumulates_labels():
    # reader receives {U} from web and {S} from crm: its context can hold {U,S}.
    res = verify(_cg(tiny()), {}, ("confidentiality",))
    w = next(w for w in res.witnesses if w.sink_tool == "web")
    assert w.label == {"U", "S"}
    assert {s.template for s in w.steps} == {"ret:web->reader", "ret:crm->reader", "call:reader->web"}


def test_memory_entries_do_not_merge():
    """Two writers store {U} and {S} separately. A trifecta gate on the *read*
    edge cannot stop the attack (each entry passes alone, the reader merges
    them); a trifecta gate on the *call* edge can."""
    d = tiny()
    d["agents"] = [
        {"id": "w1", "tools": ["web"], "memory": {"write": ["notes"]}},
        {"id": "w2", "tools": ["crm"], "memory": {"write": ["notes"]}},
        {"id": "actor", "tools": ["email"], "memory": {"read": ["notes"]}},
    ]
    cg = _cg(d)
    fixed_web = {"call:w1->web": Mode.REMOVE}  # isolate the e-mail path
    on_read = verify(cg, {**fixed_web, "mr:notes->actor": Mode.TRIFECTA}, ("confidentiality",))
    assert not on_read.safe
    on_call = verify(cg, {**fixed_web, "call:actor->email": Mode.TRIFECTA}, ("confidentiality",))
    assert on_call.safe


def test_witness_steps_are_in_dependency_order():
    res = verify(_cg(tiny()))
    for w in res.witnesses:
        produced = set()
        for s in w.steps:
            # The source of each step is a tool output or was produced earlier.
            assert s.src.startswith("out:") or s.src in produced
            produced.add(s.dst)


def test_persistent_memory_links_sessions(demo_dict):
    cg = compile_manifest(manifest_from_dict(demo_dict), sessions=3)
    persist = [(u, v) for u, v, k in cg.graph.edges(data="kind") if k == "persist"]
    assert ("mem:shared_notes@1", "mem:shared_notes@2") in persist
    assert ("mem:shared_notes@2", "mem:shared_notes@3") in persist
    assert not any(u.startswith("mem:scratchpad") for u, _ in persist)  # session memory
