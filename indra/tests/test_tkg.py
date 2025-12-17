import json
import builtins
import pytest

import indra.statements as ist
from indra.util import unicode_strs
from indra.sources import tkg


def assert_grounding_value_or_none(st):
    """Ensure there are no empty grounding values ('' or [])."""
    for a in st.agent_list():
        if a is None:
            continue
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


def test_tkg_offline_processing(tmp_path):
    proc = tkg.process_json(TEST_JSON)

    assert proc is not None
    assert hasattr(proc, "statements")

    # Expect ~3 valid BELs (invalid BEL should be skipped)
    assert len(proc.statements) == 3

    for st in proc.statements:
        assert st.evidence, "Evidence must exist"
        assert_grounding_value_or_none(st)

    found_activation = any(isinstance(s, ist.Activation) for s in proc.statements)
    found_inhibition = any(isinstance(s, ist.DecreaseAmount) for s in proc.statements)
    assert found_activation, "Expected Activation statement"
    assert found_inhibition, "Expected DecreaseAmount statement"


# V2 TESTS
def test_v2_mock(monkeypatch):
    """V2 mock test — validates the Python-wrapper integration
    WITHOUT requiring the external textToKnowledgeGraph package.
    """

    # FAKE TKG main()
    def mock_tkg_main(api_key, pmc_ids, upload_to_ndex=False, **kwargs):
        assert api_key == "FAKE_KEY"
        assert pmc_ids == ["PMC123456"]

        return {
            "relations": [
                {
                    "bel": "p(HGNC:TP53) increases p(HGNC:MDM2)",
                    "text": "TP53 activates MDM2",
                    "confidence": 0.88,
                    "pmcid": "PMC123456",
                }
            ]
        }

    # Patch import
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "textToKnowledgeGraph":
            FakeModule = type("FakeModule", (), {"main": mock_tkg_main})
            return FakeModule
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Run V2 live wrapper
    proc = tkg.process_pmc("PMC123456", api_key="FAKE_KEY")

    assert isinstance(proc, tkg.TkgProcessor)
    assert len(proc.statements) == 1

    st = proc.statements[0]

    assert any(ev.annotations.get("confidence") == 0.88 for ev in st.evidence)
    assert any(ev.annotations.get("pmcid") == "PMC123456" for ev in st.evidence)


# OPTIONAL LIVE TEST — only runs if dependencies exist
@pytest.mark.slow
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("textToKnowledgeGraph") is None,
    reason="textToKnowledgeGraph package not installed",
)
@pytest.mark.skipif(
    "OPENAI_API_KEY" not in __import__("os").environ,
    reason="OPENAI_API_KEY environment variable is not set",
)
def test_v2_live_textkg():
    """Optional real call to textToKnowledgeGraph (slow)."""
    import os

    api_key = os.environ["OPENAI_API_KEY"]
    proc = tkg.process_pmc("PMC3898398", api_key=api_key)

    assert isinstance(proc, tkg.TkgProcessor)
    assert len(proc.statements) > 0
    assert any(
        "confidence" in ev.annotations
        for st in proc.statements
        for ev in st.evidence
    )
