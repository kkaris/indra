import logging
from typing import Any, Dict, List, Optional, Sequence

from indra.statements import Evidence
from indra.sources.bel import process_bel_stmt

from .normalizer import prepare_bel_for_parsing

logger = logging.getLogger(__name__)


class LlmBelProcessor:
    """Process LLM-produced BEL JSON into INDRA Statements.

    Parameters
    ----------
    llm_results : Sequence[dict]
        Sequence of dicts in the llm_results.json format, each typically like:
        {
          "pmcid": "PMC3898398",
          "pmid": "12345678",              # optional
          "section": "Results",           # optional
          "sentence": "SIRT1 inhibits PARP1 activity.",
          "bel_statements": [
              "p(HGNC:SIRT1) decreases act(p(HGNC:PARP1))",
              ...
          ]
        }
    """

    def __init__(self, llm_results: Sequence[Dict[str, Any]]):
        self.llm_results: Sequence[Dict[str, Any]] = llm_results
        self.statements: List[Any] = []
        self._n_ok = 0
        self._n_skipped = 0
        self._n_error = 0


    # Public API
    def extract_statements(self) -> None:
        """Run BEL → PyBEL → INDRA pipeline on all LLM extractions.

        Populates self.statements with INDRA Statements and
        attaches Evidence(source_api="llm_bel", annotations=...).
        """
        for entry in self.llm_results:
            self._process_entry(entry)

        logger.info(
            "LlmBelProcessor: OK=%d, skipped=%d, errors=%d, total=%d",
            self._n_ok,
            self._n_skipped,
            self._n_error,
            len(self.llm_results),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_entry(self, entry: Dict[str, Any]) -> None:
        bel_list: Sequence[str] = entry.get("bel_statements") or []
        if not bel_list:
            self._n_skipped += 1
            return

        sentence: Optional[str] = (
            entry.get("sentence")
            or entry.get("text")
            or entry.get("summary")
        )
        pmcid: Optional[str] = entry.get("pmcid")
        pmid: Optional[str] = entry.get("pmid")
        section: Optional[str] = entry.get("section")
        model: Optional[str] = entry.get("model")
        conf: Optional[float] = entry.get("confidence")

        for raw_bel in bel_list:
            if not raw_bel:
                continue

            should_process, bel_clean = prepare_bel_for_parsing(raw_bel)
            if not should_process or not bel_clean:
                self._n_skipped += 1
                continue

            try:
                # We always want the PybelProcessor, not a single Statement.
                bp = process_bel_stmt(bel_clean, squeeze=False)
            except Exception as e:
                logger.warning(
                    "Error parsing BEL from LLM: %s [BEL=%r]", e, bel_clean
                )
                self._n_error += 1
                continue

            stmts = getattr(bp, "statements", [])
            if not stmts:
                # syntactically OK but no INDRA semantics
                self._n_skipped += 1
                continue

            for st in stmts:
                ev = Evidence(
                    source_api="llm_bel",
                    pmid=pmid,
                    text=sentence,
                    annotations={
                        "pmcid": pmcid,
                        "section": section,
                        "bel": bel_clean,
                        "raw_bel": raw_bel,
                        "llm_model": model,
                        "llm_confidence": conf,
                    },
                )
                st.evidence.append(ev)
                self.statements.append(st)
                self._n_ok += 1
