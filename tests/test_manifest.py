import pytest

from amps.labels import Mode
from amps.manifest import load_manifest, manifest_from_dict

from conftest import tiny


def test_demo_manifest_loads(demo_path):
    m = load_manifest(demo_path)
    assert m.system == "demo-4agents"
    assert len(m.agents) == 4
    edges = m.edge_templates()
    for e in ["ret:read_inbox->retrieval", "call:executor->send_email",
              "msg:supervisor->executor", "msg:executor->supervisor",
              "mw:retrieval->shared_notes", "mr:scratchpad->executor", "persist:shared_notes"]:
        assert e in edges
    # kb_search is local and trusted: it returns data but is not a sink.
    assert "call:retrieval->kb_search" not in edges


def test_output_labels(demo_path):
    m = load_manifest(demo_path)
    assert m.output_label("web_search") == {"U"}
    assert m.output_label("read_inbox") == {"U", "S"}
    assert m.output_label("crm_lookup") == {"S"}
    assert m.output_label("kb_search") == frozenset()   # internal is not "S" by default
    assert m.output_label("send_email") == frozenset()  # no output


def test_unknown_reference_is_rejected():
    d = tiny()
    d["agents"][0]["tools"].append("does_not_exist")
    with pytest.raises(ValueError, match="unknown tool"):
        manifest_from_dict(d)


def test_duplicate_id_is_rejected():
    d = tiny()
    d["memory"].append({"id": "web"})  # same id as a tool
    with pytest.raises(ValueError, match="duplicate id"):
        manifest_from_dict(d)


def test_unknown_edge_in_task_is_rejected():
    d = tiny(tasks=[{"id": "t", "requires": [{"edge": "call:reader->email"}]}])
    with pytest.raises(ValueError, match="unknown edge"):
        manifest_from_dict(d)


def test_typo_in_field_is_rejected():
    d = tiny()
    d["tools"][0]["integrity_critcal"] = True  # misspelt on purpose
    with pytest.raises(ValueError):
        manifest_from_dict(d)


def test_mode_weights_must_be_ordered():
    d = tiny(controls={"mode_weights": {"R": 0.1, "I": 0.5, "C": 0.2}})
    with pytest.raises(ValueError, match="R >= I >= C"):
        manifest_from_dict(d)


def test_descriptor_hash_detects_change():
    m1 = manifest_from_dict(tiny())
    d = tiny()
    d["tools"][1]["description"] = "now also exports records"
    m2 = manifest_from_dict(d)
    assert m1.tool("crm").descriptor_sha256() != m2.tool("crm").descriptor_sha256()
    assert m1.tool("web").descriptor_sha256() == m2.tool("web").descriptor_sha256()


def test_edge_cost_uses_weights_and_overrides():
    d = tiny(controls={"overrides": {"call:actor->email": 10}})
    m = manifest_from_dict(d)
    assert m.edge_cost("call:actor->email", Mode.REMOVE) == 10
    assert m.edge_cost("call:actor->email", Mode.INTEGRITY) == pytest.approx(6)
    assert m.edge_cost("msg:reader->actor", Mode.TRIFECTA) == pytest.approx(0.6)
