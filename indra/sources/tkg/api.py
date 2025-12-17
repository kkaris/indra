__all__ = ["process_json_file", "process_json_folder", "process_json",
           "process_pmc"]
"""
This module implements an API for the textToKnowledgeGraph
method which extracts BEL statements from publications via an LLM.

This module provides two integration modes:

Offline processing
    In this mode, a JSON file output from textToKnowledgeGraph
    is used as the starting point from which INDRA Statements are produced.

Live processing
    If the `texttoknowledgegraph` package is installed, calls
    the LLM extraction pipeline, processes the returned BEL relations
    and produces INDRA Statements.

Both modes produce an LlmBelProcessor instance containing INDRA
Statements derived from BEL expressions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Union

from .processor import LlmBelProcessor

logger = logging.getLogger(__name__)



def process_json_file(path: Union[str, Path]) -> LlmBelProcessor:
    """Process a single textToKnowledgeGraph JSON results file.

    Parameters
    ----------
    path : str or Path
        Path to a JSON file containing BEL relations.

    Returns
    -------
    LlmBelProcessor
        Processor containing the converted INDRA Statements.
    """
    path = Path(path)
    logger.debug("Processing LLM-BEL results file: %s", path)

    with open(path, "r") as fh:
        data = json.load(fh)

    return process_json(data)


def process_json_folder(path: Union[str, Path]) -> LlmBelProcessor:
    """Process all JSON files in a directory of textToKnowledgeGraph outputs.

    Parameters
    ----------
    path : str or Path
        Directory containing multiple JSON output files.

    Returns
    -------
    LlmBelProcessor
        Processor containing the union of all INDRA Statements.
    """
    processor = LlmBelProcessor()
    path = Path(path)

    for f in sorted(path.glob("*.json")):
        logger.info("Processing file: %s", f)
        with open(f, "r") as fh:
            data = json.load(f)
        sub = process_json(data)
        processor.statements.extend(sub.statements)

    return processor


def process_json(data: Dict) -> LlmBelProcessor:
    """Process BEL relations returned directly from the LLM engine.

    Parameters
    ----------
    data : dict
        Dictionary containing at least a ``"relations"`` field.

    Returns
    -------
    LlmBelProcessor
        Processor with INDRA Statements derived from BEL.
    """
    processor = LlmBelProcessor(data)
    processor.extract_statements()
    return processor


def process_pmc(pmc_id: str, api_key: str, **kwargs) -> \
        Union[LlmBelProcessor, None]:
    """Run live BEL extraction using textToKnowledgeGraph, if installed.

    Parameters
    ----------
    pmc_id : str
        PMCID such as 'PMC3898398'.
    api_key : str
        OpenAI API key required by the external textToKnowledgeGraph package.
    kwargs :
        Additional keyword arguments passed to textToKnowledgeGraph.main().

    Returns
    -------
    LlmBelProcessor
        Processor containing INDRA Statements derived from live BEL output.

    Raises
    ------
    ImportError
        If textToKnowledgeGraph is not installed.
    ValueError
        If the returned data structure is unexpected.
    """
    try:
        from texttoknowledgegraph import main as tkg_main
    except ImportError:
        raise ImportError(
            "The 'textToKnowledgeGraph' package is not installed. "
            "Install it or run textToKnowledge graph separately to "
            "produce output files and then use one of the functions like "
            "process_json_file to process the outputs."
        )

    logger.debug("Running live textToKnowledgeGraph extraction for %s", pmc_id)

    results = tkg_main(
        api_key=api_key,
        pmc_ids=[pmc_id],
        upload_to_ndex=False,
        **kwargs,
    )

    if not isinstance(results, dict) or "relations" not in results:
        logger.error(f"textToKnowledgeGraph returned unexpected "
                     f"structure: {type(results)}")
        return None

    return process_json(results)
