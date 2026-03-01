"""
src/semantic_index.py
Semantic expansion tables and tokenisation helpers for multimodal scene search.

Contains:
  - SEMANTIC_MAP  — token → related tokens (object synonyms)
  - INTENT_MAP    — natural-language intent → internal scene labels
  - _FILLER_TOKENS — tokens that carry no direct scene signal
  - _tokenise()   — shared tokenisation helper
  - _overlap_score() — token overlap scoring helper
"""

import re

# ── Semantic expansion map ────────────────────────────────────────────────────
# Maps a query token to a list of related tokens/phrases that should also match.
# Deterministic, zero-dependency, fully explainable.
SEMANTIC_MAP: dict[str, list[str]] = {
    "plant":       ["potted plant", "flower"],
    "flower":      ["potted plant"],
    "car":         ["vehicle", "truck"],
    "vehicle":     ["car", "truck"],
    "drink":       ["drinking", "bottle"],
    "talk":        ["conversation"],
    "celebrating": ["celebration", "running"],
}

# ── Intent map ────────────────────────────────────────────────────────────────
# Maps natural-language intent tokens to the internal tags / object labels that
# represent them in scene data.
INTENT_MAP: dict[str, list[str]] = {
    # Actor normalisation
    "woman":        ["person"],
    "man":          ["person"],
    "girl":         ["person"],
    "boy":          ["person"],
    "people":       ["person"],
    # Sports / trophy
    "world cup":    ["trophy"],
    "cup":          ["trophy"],
    "trophy":       ["trophy"],
    "goal":         ["sports ball"],
    # Driving
    "drive":        ["vehicle_moving", "car"],
    "driving":      ["vehicle_moving", "car"],
    # Motion / movement
    "run":          ["running"],
    "running":      ["running"],
    "walk":         ["walking"],
    "walking":      ["walking"],
    # Dialogue
    "talk":         ["conversation"],
    "talking":      ["conversation"],
    "conversation": ["conversation"],
    # Celebration
    "celebrate":    ["fast_movement"],
    "celebration":  ["fast_movement", "running"],
    "celebrating":  ["fast_movement", "running"],
    "chase":        ["running", "vehicle_moving"],
    "moment":       [],   # neutral — only expands other tokens
    "scene":        [],   # neutral — only expands other tokens
    "player":       ["person", "sports ball", "running"],
}

# Common filler / number words that modify intent but carry no direct scene signal.
_FILLER_TOKENS: frozenset[str] = frozenset({
    "two", "one", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "some", "few", "lot", "many", "get", "got", "getting",
    "with", "show", "where", "that", "this", "the", "and", "for", "are",
    "was", "has", "have", "their", "from", "into", "onto",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    """Lowercase alphabetic tokens longer than 2 characters."""
    return {w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) > 2}


def _overlap_score(query_tokens: set[str], target_text: str) -> float:
    """
    Fraction of query tokens found in target text.
    Returns value in [0, 1].
    """
    if not query_tokens or not target_text:
        return 0.0
    target_tokens = _tokenise(target_text)
    matched = query_tokens & target_tokens
    return len(matched) / len(query_tokens)
