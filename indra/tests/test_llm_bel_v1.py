import json
from pathlib import Path
import indra.statements as ist
from indra.util import unicode_strs
from indra.sources import llm_bel

def assert_if_hgnc_then_up(st):
    """If an agent has an HGNC grounding, ensure UP grounding exists."""
    for a in st.agent_list():
        if a is None:
            continue
        up = a.db_refs.get("UP")
        hgnc = a.db_refs.get("HGNC")
        if hgnc and not up:
            assert False, f"Agent {a.name} has HGNC={hgnc} but no UP grounding"


def assert_grounding_value_or_none(st):
    """Ensure there are no empty grounding values ("" or [])."""
    for a in st.agent_list():
        if a is None:
            continue
        for k, v in a.db_refs.items():
            if not v:
                assert v is None, f"Invalid grounding value {k}={v}"


TEST_JSON = {
    "relations": [
        {
            "bel": "p(HGNC:SIRT1) increases act(p(HGNC:PARP1))",
            "evidence": "SIRT1 activates PARP1",
            "confidence": 0.92,
        },
        {
            "bel": "p(HGNC:SIRT1) decreases p(HGNC:MYC)",
            "evidence": "SIRT1 represses MYC",
            "confidence": 0.88,
        },
        {
            # SHOULD FAIL (invalid BEL syntax → no statements)
            "bel": "p(HGNC:SIRT1) increases act(p(HGNC:PARP1)",
            "evidence": "Malformed BEL",
            "confidence": 0.30,
        },
        {
            # Semi-supported BEL → should produce statements
            "bel": "p(FPLX:ERK) directlyIncreases act(p(HGNC:PARP1))",
            "evidence": "ERK activates PARP1",
            "confidence": 0.91,
        },
    ]
}


def test_llm_bel_offline_processing(tmp_path):
    """Test that a medium-size BEL JSON properly produces INDRA Statements."""

    # Write test JSON to a file (offline V1 behavior)
    json_path = tmp_path / "llm_results.json"
    json_path.write_text(json.dumps(TEST_JSON), encoding="utf-8")

    # Run V1 ingestion pipeline
    proc = llm_bel.process_llm_results_file(json_path)

    assert proc is not None
    assert hasattr(proc, "statements")

    # Expect:
    #   - 3 valid BELs → ~3 statement groups
    #   - malformed BEL → 0 statements
    assert len(proc.statements) >= 3

    for st in proc.statements:
        assert st.evidence, "Evidence must exist"
        assert unicode_strs((st,)), "Unicode safety failed"
        assert_grounding_value_or_none(st)
        assert_if_hgnc_then_up(st)

    # Check specific types exist
    found_activation = any(isinstance(s, ist.Activation) for s in proc.statements)
    found_inhibition = any(isinstance(s, ist.DecreaseAmount) for s in proc.statements)
    assert found_activation, "Expected Activation statement"
    assert found_inhibition, "Expected DecreaseAmount statement"

    # Metadata propagation check
    for st in proc.statements:
        has_confidence = any(
            ("confidence" in ev.annotations) and (ev.annotations["confidence"] is not None)
            for ev in st.evidence
        )
        assert has_confidence, "Missing confidence metadata"
