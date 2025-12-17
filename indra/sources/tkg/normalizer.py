import re

# Relations that INDRA/PyBEL does NOT support in V1/V2
UNSUPPORTED_RELATIONS = [
    "hasActivity",
    "hasComponent",
    "rateLimitingStepOf",
    "positiveCorrelation",
    "negativeCorrelation",
]


def is_supported_bel(bel: str) -> bool:
    """Return False if the BEL string contains unsupported relation types."""
    bel = bel.strip()
    return not any(rel in bel for rel in UNSUPPORTED_RELATIONS)


# 1. Fix GO Biological Process names that contain spaces
GO_BP_PATTERN = re.compile(r'GO:([A-Za-z0-9\-\s]+)')


def normalize_go_terms(bel: str) -> str:
    """Normalize GO terms like:
        GO:DNA-templated transcription
    into:
        GO:"DNA-templated transcription"
    so PyBEL can parse them.
    """
    def replacer(match):
        content = match.group(1)
        # If already quoted, do nothing
        if '"' in content or "'" in content:
            return f'GO:{content}'
        return f'GO:"{content}"'

    return GO_BP_PATTERN.sub(replacer, bel)


# 2. Remove stray spaces and malformed parentheses
def normalize_parentheses(bel: str) -> str:
    """Fix basic spacing and parenthesis issues seen in LLM BEL output."""
    bel = bel.replace("( ", "(").replace(" )", ")")
    # Collapse double spaces once – cheap and good enough
    bel = bel.replace("  ", " ")
    bel = bel.strip()
    return bel


# 3. Fix CHEBI formatting inconsistencies
def normalize_chebi(bel: str) -> str:
    """Normalize CHEBI names and trailing spaces."""
    # a(CHEBI:"NAD(+)" )
    bel = re.sub(r'CHEBI:"([^"]+)"\s*\)', r'CHEBI:"\1")', bel)
    # a(CHEBI:NAD)
    bel = re.sub(r'CHEBI:([A-Za-z0-9\-\+]+)\)', r'CHEBI:\1)', bel)
    return bel



# 4. Combined entry point for all normalization
def normalize_bel(bel: str) -> str:
    """Apply all normalization steps."""
    bel = bel.strip()
    bel = normalize_parentheses(bel)
    bel = normalize_go_terms(bel)
    bel = normalize_chebi(bel)
    return bel


# 5. Final helper used by processor/api
def prepare_bel_for_parsing(bel: str):
    """Return (should_process: bool, cleaned_bel: str or None)."""
    if not is_supported_bel(bel):
        return False, None
    cleaned = normalize_bel(bel)
    return True, cleaned
