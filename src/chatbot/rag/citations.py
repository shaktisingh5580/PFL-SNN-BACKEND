"""
Legal citation extraction and formatting.
Extracts, deduplicates, and formats legal citations from retrieved regulatory chunks.
"""
import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def extract_citations(text: str) -> List[Dict]:
    """
    Extract legal citations from text.

    Patterns matched:
    - "Section X.Y" / "§ X.Y"
    - "Gujarat GDCR 2017"
    - "Gujarat TP & UD Act 1976"
    - "SUDA Development Plan 2035"
    """
    citations = []

    # Section references
    section_pattern = re.compile(
        r'(?:Section|§)\s*(\d+(?:\.\d+)*)',
        re.IGNORECASE
    )
    for match in section_pattern.finditer(text):
        section = match.group(1)
        # Try to find the act name nearby
        context_start = max(0, match.start() - 100)
        context = text[context_start:match.end() + 50]
        act_name = _find_act_in_context(context)

        citations.append({
            "section": section,
            "act_name": act_name,
            "full_citation": f"{act_name}, Section {section}" if act_name else f"Section {section}",
            "start_pos": match.start(),
        })

    # Act name references without sections
    act_patterns = [
        (r'Gujarat\s+GDCR\s+2017', "Gujarat GDCR 2017"),
        (r'Gujarat\s+TP\s*&?\s*UD\s+Act\s*,?\s*1976', "Gujarat TP & UD Act 1976"),
        (r'SUDA\s+Development\s+Plan\s+2035', "SUDA Development Plan 2035"),
        (r'Gujarat\s+Town\s+Planning.*?Act.*?1976', "Gujarat TP & UD Act 1976"),
        (r'Environmental\s+Protection\s+Act', "Environmental Protection Act"),
    ]

    for pattern, name in act_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            # Check if we already have a section citation for this pos
            if not any(abs(c["start_pos"] - match.start()) < 20 for c in citations):
                citations.append({
                    "section": None,
                    "act_name": name,
                    "full_citation": name,
                    "start_pos": match.start(),
                })

    return citations


def deduplicate_citations(citations: List[Dict]) -> List[Dict]:
    """Remove duplicate citations, keeping the most complete version."""
    seen = set()
    unique = []

    for c in citations:
        key = (c.get("act_name", ""), c.get("section", ""))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def format_citation(citation: Dict, style: str = "full") -> str:
    """
    Format a citation in the specified style.

    Styles:
        - "full": "Gujarat GDCR 2017, Section 12.3, Page 47"
        - "short": "GDCR § 12.3"
        - "footnote": "[1] Gujarat GDCR 2017, Section 12.3"
    """
    act = citation.get("act_name", "Unknown Act")
    section = citation.get("section")
    page = citation.get("page_number")

    if style == "short":
        short_name = act.replace("Gujarat ", "").replace("Development ", "Dev. ")
        if section:
            return f"{short_name} § {section}"
        return short_name

    elif style == "footnote":
        idx = citation.get("index", 1)
        parts = [f"[{idx}] {act}"]
        if section:
            parts.append(f"Section {section}")
        if page:
            parts.append(f"Page {page}")
        return ", ".join(parts)

    else:  # "full"
        parts = [act]
        if section:
            parts.append(f"Section {section}")
        if page:
            parts.append(f"Page {page}")
        return ", ".join(parts)


def format_citations_block(retrieved_chunks: List[Dict]) -> str:
    """
    Extract and format all citations from retrieved chunks
    into a formatted citations block for bot responses.
    """
    all_citations = []

    for chunk in retrieved_chunks:
        text = chunk.get("content", "")
        metadata = chunk.get("metadata", {})

        citations = extract_citations(text)
        for c in citations:
            c["page_number"] = metadata.get("page_number")
            c["source_file"] = metadata.get("source_file")
        all_citations.extend(citations)

    unique = deduplicate_citations(all_citations)

    if not unique:
        return ""

    lines = ["\n📚 **Legal References:**"]
    for i, c in enumerate(unique, 1):
        c["index"] = i
        lines.append(f"  {format_citation(c, 'footnote')}")

    return "\n".join(lines)


def _find_act_in_context(context: str) -> str:
    """Find the act name nearest to a section reference."""
    acts = [
        ("Gujarat GDCR 2017", r'GDCR\s*2017'),
        ("Gujarat TP & UD Act 1976", r'TP\s*&?\s*UD\s+Act'),
        ("SUDA Development Plan 2035", r'SUDA.*?2035'),
        ("Gujarat GDCR 2017", r'Gujarat\s+General\s+Development'),
    ]
    for name, pattern in acts:
        if re.search(pattern, context, re.IGNORECASE):
            return name
    return "Gujarat GDCR 2017"  # Default
