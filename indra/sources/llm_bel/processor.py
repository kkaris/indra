import logging
from typing import Any, Dict, List, Optional, Sequence

from indra.statements import Evidence
from indra.sources.bel import process_bel_stmt
from .normalizer import prepare_bel_for_parsing

logger = logging.getLogger(__name__)


class LlmBelProcessor:
    """
    Processor that handles BEL relations extracted from an LLM.

    After parsing BEL → INDRA Statements via PyBEL, this processor attaches
    LLM metadata (confidence, text, pmid, pmcid, etc.) to Evidence objects.
    """

    def __init__(self, llm_results: Optional[Sequence[Dict]] = None, *, source=None):
        self.llm_results = llm_results or []
        self.source = source
        self.statements: List[Any] = []

        self._n_ok = 0
        self._n_skipped = 0
        self._n_error = 0

    # Main API used by process_llm_results_json()
    def add_from_pybel_processor(self, pp, extra_metadata: Dict[str, Any]):
        """Attach PyBEL → INDRA statements with LLM metadata."""
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
        """Run BEL → INDRA pipeline for all entries in llm_results."""
        for entry in self.llm_results:
            self._process_entry(entry)

        logger.info(
            "LlmBelProcessor finished: OK=%d skipped=%d error=%d total=%d",
            self._n_ok, self._n_skipped, self._n_error, len(self.llm_results)
        )

    def _process_entry(self, entry: Dict[str, Any]) -> None:
        bel_list: Sequence[str] = entry.get("bel_statements") or []
        if not bel_list:
            self._n_skipped += 1
            return

        sentence = (
            entry.get("sentence")
            or entry.get("text")
            or entry.get("summary")
            or ""
        )
        pmcid = entry.get("pmcid")
        pmid = entry.get("pmid")
        section = entry.get("section")
        model = entry.get("model")
        conf = entry.get("confidence")

        for raw_bel in bel_list:

            # BEL normalization
            should_process, bel_clean = prepare_bel_for_parsing(raw_bel)
            if not should_process or not bel_clean:
                self._n_skipped += 1
                continue

            try:
                pp = process_bel_stmt(bel_clean)
            except Exception as e:
                logger.warning(f"BEL parse failed: {e} [BEL={bel_clean}]")
                self._n_error += 1
                continue

            if not getattr(pp, "statements", []):
                self._n_skipped += 1
                continue

            # Attach evidence + metadata
            for st in pp.statements:
                ev = Evidence(
                    source_api="llm_bel",
                    text=sentence,
                    pmid=pmid,
                    annotations={
                        "bel": bel_clean,
                        "raw_bel": raw_bel,
                        "pmcid": pmcid,
                        "section": section,
                        "confidence": conf,
                        "llm_model": model,
                    },
                )
                st.evidence.append(ev)
                self.statements.append(st)
                self._n_ok += 1
