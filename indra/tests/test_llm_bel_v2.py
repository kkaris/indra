import pytest
import builtins

from indra.sources.llm_bel.api import process_pmc_live
from indra.sources.llm_bel.processor import LlmBelProcessor


def test_v2_mock(monkeypatch):
    """V2 mock test — validates the Python-wrapper integration
    WITHOUT requiring the external textToKnowledgeGraph package."""

    # Mock TKG .main() output
    def mock_tkg_main(api_key, pmc_ids, upload_to_ndex=False, **kwargs):
        assert api_key == "FAKE_KEY"
        assert pmc_ids == ["PMC123456"]

        # Mimic expected structure from real LLM extractor
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

    # Patch Python import so that:
    #    import textToKnowledgeGraph → returns mock_tkg_main()
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "textToKnowledgeGraph":
            FakeModule = type("FakeModule", (), {"main": mock_tkg_main})
            return FakeModule
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Call the actual V2 integration function
    proc = process_pmc_live("PMC123456", api_key="FAKE_KEY")

    # 4. Assertions — deep verification of correctness
    assert isinstance(proc, LlmBelProcessor)
    assert len(proc.statements) == 1

    st = proc.statements[0]

    # Evidence ordering is not guaranteed — check all of them.
    has_conf = any(ev.annotations.get("confidence") == 0.88 for ev in st.evidence)
    assert has_conf, "LLM confidence should have been propagated"

    # pmcid propagation
    has_pmcid = any(ev.annotations.get("pmcid") == "PMC123456" for ev in st.evidence)
    assert has_pmcid, "PMCID metadata missing"



# OPTIONAL LIVE TEST — REQUIRES textToKnowledgeGraph + API KEY
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
    """
    OPTIONAL end-to-end live test.

    Requires:
    - pip install textToKnowledgeGraph
    - export OPENAI_API_KEY="sk-..."

    This exercises the REAL LLM extraction pipeline.
    """

    import os
    api_key = os.environ["OPENAI_API_KEY"]

    proc = process_pmc_live("PMC3898398", api_key=api_key)

    assert isinstance(proc, LlmBelProcessor)
    assert len(proc.statements) > 0, "Live LLM extractor returned no statements"

    # Ensure at least one confidence annotation is present
    assert any(
        "confidence" in ev.annotations for st in proc.statements for ev in st.evidence
    )
