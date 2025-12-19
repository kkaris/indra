import indra.statements as ist
from indra.sources import tkg


def assert_grounding_value_or_none(stmt):
    """Ensure there are no empty grounding values ('' or [])."""
    for a in stmt.real_agent_list():
        for k, v in a.db_refs.items():
            if not v:
                assert v is None, f"Invalid grounding value {k}={v}"


TEST_JSON = {
    "LLM_extractions": [
        {
            "Results": [
                {
                    "bel_statement": "p(HGNC:SIRT1) increases act(p(HGNC:PARP1))",
                    "evidence": "SIRT1 activates PARP1",
                },
                {
                    "bel_statement": "p(HGNC:SIRT1) decreases p(HGNC:MYC)",
                    "evidence": "SIRT1 represses MYC",
                },
                {
                    "bel_statement": "p(HGNC:SIRT1) increases act(p(HGNC:PARP1)",
                    "evidence": "Malformed BEL",
                },
                {
                    "bel_statement": "p(FPLX:ERK) directlyIncreases act(p(HGNC:PARP1))",
                    "evidence": "ERK activates PARP1",
                },
            ]
        }
    ]
}


def test_tkg_processing(tmp_path):
    proc = tkg.process_json(TEST_JSON)

    assert proc is not None
    assert hasattr(proc, "statements")

    # Expect 3 valid BELs (invalid BEL should be skipped)
    assert len(proc.statements) == 3

    for st in proc.statements:
        assert st.evidence, "Evidence must exist"
        assert_grounding_value_or_none(st)

    assert any(isinstance(s, ist.Activation) for s in proc.statements)
    assert any(isinstance(s, ist.DecreaseAmount) for s in proc.statements)
