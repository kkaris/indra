"""
LLM-BEL INDRA Source

This module exposes the public API for integrating BEL statements generated
by an LLM extraction pipeline into INDRA.

Exports:
    • process_text
    • process_text_remote
    • process_llm_results_file
    • process_llm_results_dir
    • process_llm_results_json
    • LlmBelProcessor
    • prepare_bel_for_parsing
"""

from .api import (
    process_text,
    process_text_remote,
    process_llm_results_file,
    process_llm_results_dir,
    process_llm_results_json,
)

from .processor import LlmBelProcessor
from .normalizer import prepare_bel_for_parsing

__all__ = [
    "process_text",
    "process_text_remote",
    "process_llm_results_file",
    "process_llm_results_dir",
    "process_llm_results_json",
    "LlmBelProcessor",
    "prepare_bel_for_parsing",
]
