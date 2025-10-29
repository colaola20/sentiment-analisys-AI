from __future__ import annotations

import re
from typing import Iterable, List


_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MARKDOWN_LINK_PATTERN = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<url>[^\)]+)\)")
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean text for sentiment analysis.

    - Remove markdown links (keep anchor text)
    - Remove raw URLs
    - Strip HTML-like tags
    - Normalize whitespace
    - Keep basic punctuation
    """
    if not text:
        return ""
    s = text.replace("\u200b", " ").replace("\xA0", " ")
    # Replace markdown links with just the text
    s = _MARKDOWN_LINK_PATTERN.sub(lambda m: m.group("text"), s)
    # Remove URLs
    s = _URL_PATTERN.sub(" ", s)
    # Remove HTML tags
    s = _HTML_TAG_PATTERN.sub(" ", s)
    # Normalize whitespace
    s = _MULTI_SPACE_PATTERN.sub(" ", s)
    s = s.strip()
    return s


def batch_clean_text(texts: Iterable[str]) -> List[str]:
    return [clean_text(t) for t in texts]


