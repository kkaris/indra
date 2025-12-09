"""
LLM-BEL Source API for INDRA

This module exposes a high-level API for converting BEL-like relations
produced by the LLM extraction engine into INDRA Statements.

Two usage modes are supported:

1. Offline processing (V1)
   - User provides a pre-generated llm_results.json file.
   - We normalize BEL syntax, validate expressions, and convert any
     supported BEL statement into INDRA Statements using the standard
     PyBEL → PybelProcessor → INDRA pipeline.

2. Live integration via API wrapper (V2)
   - INDRA can call a running LLM extraction service.
   - The service returns BEL statements in JSON, which flow directly
     through the same processing pipeline as V1.
   - This mirrors the design of TRIPS/REACH “process_text()” interfaces.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union

from indra.sources.bel import process_bel_stmt
from .processor import LlmBelProcessor
from .normalizer import prepare_bel_for_parsing


# LLM API CLIENT (V2)
class HttpLlmBelApiClient:
    """
    Lightweight HTTP client for calling the LLM extraction service.

    This wrapper keeps V2 self-contained:
    - No external dependencies beyond 'requests'
    - Simple, explicit API similar to TRIPS/REACH processors
    - Can be swapped for another backend without touching INDRA code

    Expected remote API contract:
        POST /extract
        { "text": "...", "paper_id": optional }

    Returns:
        { "relations": [ { "bel": "...", ... }, ... ] }
    """
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def extract(self, text: str) -> List[Dict]:
        """Send raw text to the extraction API and return BEL records."""
        import requests

        payload = {"text": text}
        url = f"{self.base_url}/extract"

        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("relations", [])



# HIGH-LEVEL ENTRY POINTS
def process_text(text: str):
    """
    Convert a raw BEL string (manually supplied) into INDRA statements.

    This function is *not* intended for normal usage; it exists so that
    LLM-BEL behaves like other INDRA sources that implement process_text().

    Example:
        process_text("p(HGNC:SIRT1) increases act(p(HGNC:PARP1))")
    """
    bel = prepare_bel_for_parsing(text)
    pp = process_bel_stmt(bel)

    return LlmBelProcessor.from_bel_processor(pp, metadata={"source": "manual_text"})


def process_text_remote(text: str, client: "HttpLlmBelApiClient"):
    """
    High-level V2 function that:
        1. Sends text to the LLM extraction server
        2. Receives BEL-like relations
        3. Passes them through the INDRA BEL → Statement pipeline

    This enables real-time extraction, allowing users to do:

        from indra.sources import llm_bel
        client = llm_bel.HttpLlmBelApiClient("http://localhost:8000")
        proc = llm_bel.process_text_remote("SIRT1 inhibits PARP1", client)

    Returns:
        LlmBelProcessor with populated INDRA Statements
    """
    relations = client.extract(text)
    return process_llm_results_json({"relations": relations})


# OFFLINE PROCESSING (V1)
def process_llm_results_file(path: Union[str, Path]):
    """
    Load a single llm_results.json file and convert all BEL statements.

    This enables offline processing of a paper's extracted relations
    without needing a running LLM inference server.
    """
    with open(path, "r") as fh:
        data = json.load(fh)
    return process_llm_results_json(data, source_path=str(path))


def process_llm_results_dir(path: Union[str, Path]):
    """
    Process all llm_results.json files in a directory.

    Useful when the extraction repo stores:
        paper_id/llm_results.json
    """
    path = Path(path)
    processors = []

    for file in path.rglob("llm_results.json"):
        processors.append(process_llm_results_file(file))

    return processors


def process_llm_results_json(data: Dict, source_path: Optional[str] = None):
    """
    Convert an LLM extraction JSON structure into INDRA Statements.

    Expected JSON format:
        {
            "relations": [
                {
                    "bel": "p(HGNC:SIRT1) increases act(p(HGNC:PARP1))",
                    "text": "... optional ...",
                    "confidence": 0.95,
                    ...
                },
                ...
            ]
        }

    Pipeline:
        - Normalize BEL string (fix GO label formatting, quotes, spaces)
        - Validate using PyBEL
        - Convert to INDRA Statements via PybelProcessor
        - Attach metadata and provenance
    """

    processor = LlmBelProcessor(source=source_path)

    for rel in data.get("relations", []):
        bel_raw = rel.get("bel")
        if not bel_raw:
            continue

        # BEL cleanup (fixing syntax issues learned from your validation)
        bel = prepare_bel_for_parsing(bel_raw)

        # Parse BEL → PyBEL → INDRA
        pp = process_bel_stmt(bel)
        stmts = pp.statements or []

        # Add metadata from the LLM output itself
        processor.add_from_pybel_processor(pp, extra_metadata=rel)

    return processor
