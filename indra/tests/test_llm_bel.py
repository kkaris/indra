"""
Tests for llm_bel V1 (offline JSON ingestion) and V2 (live wrapper)
pipelines. Both test sets are independent and validate different parts
of the LLM-BEL reader.

V1 — Offline JSON Processing
----------------------------
Ensures that static BEL relations are correctly parsed into INDRA
Statements, malformed BEL is safely skipped, grounding is valid, and
evidence metadata (confidence, text) is propagated.

V2 — Python Wrapper Integration
-------------------------------
Uses import-level mocking to simulate textToKnowledgeGraph.main() and
checks that the wrapper produces valid INDRA Statements and preserves
metadata (confidence, pmcid). No network or API key is required.

An optional live test exercises the real external extractor and is
skipped unless all dependencies and environment variables are available.
"""

import json
from pathlib import Path
import builtins
import pytest

import indra.statements as ist
from indra.util import unicode_strs
from indra.sources import llm_bel
from indra.sources.llm_bel.api import process_pmc_live
from indra.sources.llm_bel.processor import LlmBelProcessor


# Helper Assertions (shared by V1 tests)
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
    """Ensure there are no empty grounding values ('' or [])."""
    for a in st.agent_list():
        if a is None:
            continue
        for k, v in a.db_refs.items():
            if not v:
                assert v is None, f"Invalid grounding value {k}={v}"


# V1 TESTS

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
    """V1 offline processing — ensure BEL JSON → INDRA statements."""

    json_path = tmp_path / "llm_results.json"
    json_path.write_text(json.dumps(TEST_JSON), encoding="utf-8")

    proc = llm_bel.process_llm_results_file(json_path)

    assert proc is not None
    assert hasattr(proc, "statements")

    # Expect ~3 valid BELs (invalid BEL should be skipped)
    assert len(proc.statements) >= 3

    for st in proc.statements:
        assert st.evidence, "Evidence must exist"
        assert unicode_strs((st,)), "Unicode safety failed"
        assert_grounding_value_or_none(st)
        assert_if_hgnc_then_up(st)

    found_activation = any(isinstance(s, ist.Activation) for s in proc.statements)
    found_inhibition = any(isinstance(s, ist.DecreaseAmount) for s in proc.statements)
    assert found_activation, "Expected Activation statement"
    assert found_inhibition, "Expected DecreaseAmount statement"

    # Confidence propagation
    for st in proc.statements:
        assert any(
            "confidence" in ev.annotations and ev.annotations["confidence"] is not None
            for ev in st.evidence
        )


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
    proc = process_pmc_live("PMC123456", api_key="FAKE_KEY")

    assert isinstance(proc, LlmBelProcessor)
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
    proc = process_pmc_live("PMC3898398", api_key=api_key)

    assert isinstance(proc, LlmBelProcessor)
    assert len(proc.statements) > 0
    assert any(
        "confidence" in ev.annotations
        for st in proc.statements
        for ev in st.evidence
    )
