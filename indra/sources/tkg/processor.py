__all__ = ["TkgProcessor"]

import logging
from typing import Dict, List

from indra.sources.bel import process_bel_stmt
from .normalizer import normalize_bel

logger = logging.getLogger(__name__)


class TkgProcessor:
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
        self.skipped = []

    # Alternative processing mode (not used by V1 tests but available)
    def extract_statements(self):
        """Run BEL to INDRA pipeline for all entries in llm_results."""
        extractions = self.results.get('LLM_extractions', [])
        for extraction in extractions:
            results = extraction.get('Results', [])
            for entry in results:
                raw_bel_stmt = entry['bel_statement']
                bel_stmt = normalize_bel(raw_bel_stmt)
                if raw_bel_stmt != bel_stmt:
                    logger.info('%s / %s' % (raw_bel_stmt, bel_stmt))
                try:
                    pp = process_bel_stmt(bel_stmt)
                except Exception as e:
                    self.skipped.append(bel_stmt)
                    continue
                if pp and pp.statements:
                    self.statements += pp.statements
                else:
                    self.skipped.append(bel_stmt)

        logger.debug(
            "textToKnowledgeGraph processor finished: extracted=%d "
            "skipped=%d total=%d", len(self.statements), len(self.skipped),
            len(self.results)
        )
