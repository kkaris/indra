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

        self._n_skipped = 0
        self._n_error = 0

    # Alternative processing mode (not used by V1 tests but available)
    def extract_statements(self):
        """Run BEL to INDRA pipeline for all entries in llm_results."""
        extractions = self.results.get('LLM_extractions', [])
        for extraction in extractions:
            results = extraction.get('Results', [])
            for entry in results:
                try:
                    pp = process_bel_stmt(entry['bel_statement'])
                except Exception as e:
                    self._n_error += 1
                    continue
                if pp and pp.statements:
                    self.statements += pp.statements
                else:
                    self._n_skipped += 1

        logger.debug(
            "LlmBelProcessor finished: extracted=%d skipped=%d error=%d "
            "total=%d", len(self.statements), self._n_skipped, self._n_error,
            len(self.results)
        )
