import logging
from typing import Any, Dict, List, Optional, Sequence

from indra.statements import Evidence
from indra.sources.bel import process_bel_stmt
from .normalizer import prepare_bel_for_parsing

logger = logging.getLogger(__name__)


class LlmBelProcessor:
    """Processor extracting INDRA Statments from textToKnowledgeGraph output.

    After parsing BEL to INDRA Statements via PyBEL, this processor attaches
    metadata (confidence, text, pmid, pmcid, etc.) to Evidence objects.

    Parameters
    ----------
    results : Dict
        Output data structure of textToKnowledgeGraph to be processed

    Attributes
    ----------
    statements : List[indra.statements.Statement]
        A list of INDRA Statements extracted from the results.
    """

    def __init__(self, results):
        self.results = results
        self.statements = []

        self._n_ok = 0
        self._n_skipped = 0
        self._n_error = 0

    # Main API used by process_llm_results_json()
    def add_from_pybel_processor(self, pp, extra_metadata: Dict[str, Any]):
        """Attach PyBEL to INDRA statements with LLM metadata."""
        cleaned_bel = extra_metadata.get("bel")
        raw_bel = extra_metadata.get("raw_bel") or extra_metadata.get("bel")

        text = (
            extra_metadata.get("text")
            or extra_metadata.get("sentence")
            or extra_metadata.get("summary")
            or ""
        )

        for st in pp.statements or []:
            ev = Evidence(
                source_api="llm_bel",
                text=text,
                pmid=extra_metadata.get("pmid"),
                annotations={
                    "bel": cleaned_bel,
                    "raw_bel": raw_bel,  # preserve original LLM BEL
                    "confidence": extra_metadata.get("confidence"),
                    "pmcid": extra_metadata.get("pmcid"),
                    "section": extra_metadata.get("section"),
                    "llm_model": extra_metadata.get("model"),
                },
            )
            st.evidence.append(ev)
            self.statements.append(st)
            self._n_ok += 1

    # Alternative processing mode (not used by V1 tests but available)
    def extract_statements(self):
        """Run BEL to INDRA pipeline for all entries in llm_results."""
        extractions = self.results.get('LLM_extractions', [])
        for extraction in extractions:
            results = extraction.get('Results', [])
            for entry in results:
                stmt = process_bel_stmt(entry['bel_statement'])
                if stmt:
                    self.statements.append(stmt)

        logger.debug(
            "LlmBelProcessor finished: OK=%d skipped=%d error=%d total=%d",
            self._n_ok, self._n_skipped, self._n_error, len(self.results)
        )
