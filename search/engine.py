"""
search/engine.py
Multimodal semantic search over the scene-level narrative timeline.

Scoring strategy (scene-level):
  – Transcript match           (weight 0.40)
  – Dominant objects match     (weight 0.20)
  – Supporting objects match   (weight 0.10)  ← always searched
  – Main character bonus       (weight 0.20)
  – Importance score           (weight 0.10)

Results are deduplicated by scene_id so the frontend shows one card per scene.
"""

import re
import logging
from dataclasses import dataclass, field

from video_engine.event_builder import SceneNarrative

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TRANSCRIPT_WEIGHT         = 0.40
DOMINANT_OBJECT_WEIGHT    = 0.20   # high-frequency / structurally important
SUPPORTING_OBJECT_WEIGHT  = 0.10   # present but not dominant — still searchable
MAIN_CHAR_WEIGHT          = 0.20
IMPORTANCE_WEIGHT         = 0.10   # reduced so supporting objects don't get crowded out
SCORE_THRESHOLD           = 0.06   # minimum score to include in results

# ── Semantic expansion map ────────────────────────────────────────────────────
# Maps a query token to a list of related tokens/phrases that should also match.
# Deterministic, zero-dependency, fully explainable.
# ⚠️  NEVER add "person" as a value here — it would cause person-dominated
# scenes to match every query that expands into a common object class.
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
# represent them in scene data.  Used by both query expansion and the composite
# intent matcher so the two layers stay perfectly in sync.
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

# Common filler / number words that modify intent but carry no direct scene
# signal.  Tokens in this set are not required to appear in a scene for it
# to pass the per-token gate.
_FILLER_TOKENS: frozenset[str] = frozenset({
    "two", "one", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "some", "few", "lot", "many", "get", "got", "getting",
    # Prepositions / auxiliary verbs that carry no direct scene signal
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


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single scene-level search result."""
    start_time: float
    end_time: float
    scene_id: int
    score: float
    dominant_objects: list[str]
    supporting_objects: list[str]
    main_characters: list[int]
    new_character_entry: list[int]
    importance_score: float
    event_type: str
    transcript_text: str
    formatted_time: str
    matched_terms: list[str]
    explanation: str = ""
    scene_description: str = ""
    matched_description: str = ""

    def to_dict(self) -> dict:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "scene_id": self.scene_id,
            "score": round(self.score, 3),
            "dominant_objects": self.dominant_objects,
            "supporting_objects": self.supporting_objects,
            "main_characters": self.main_characters,
            "new_character_entry": self.new_character_entry,
            "importance_score": self.importance_score,
            "event_type": self.event_type,
            "transcript_text": self.transcript_text,
            "formatted_time": self.formatted_time,
            "matched_terms": self.matched_terms,
            "explanation": self.explanation,
            "scene_description": self.scene_description,
            "matched_description": self.matched_description,
        }


# ── Composite intent matcher ─────────────────────────────────────────────────

def scene_matches_composite_intent(
    scene: "SceneNarrative",
    expanded_tokens: set[str],
) -> bool | None:
    """
    Evaluate composite query intent against a scene.

    Returns:
        True  — scene satisfies the composite intent (always include).
        False — composite intent applies but scene fails its conditions (exclude).
        None  — no specific composite intent detected; fall through to standard
                token matching.
    """
    objects_lc   = [o.lower() for o in scene.dominant_objects + scene.supporting_objects]
    action_tags  = [t.lower() for t in getattr(scene, "action_tags", [])]
    transcript   = scene.transcript_summary.lower()
    motion       = getattr(scene, "motion_intensity", 0.0)

    # ── Driving scene: person + vehicle + vehicle_moving action ──────────────
    if "vehicle_moving" in expanded_tokens:
        has_person  = any("person" in o for o in objects_lc)
        has_vehicle = any(o in ["car", "truck", "bus", "vehicle"] for o in objects_lc)
        is_moving   = "vehicle_moving" in action_tags
        return has_person and has_vehicle and is_moving

    # ── Trophy / world-cup moment ─────────────────────────────────────────────
    if "trophy" in expanded_tokens:
        has_trophy     = any("trophy" in o for o in objects_lc)
        high_imp       = scene.importance_score > 0.6
        high_motion    = motion > 1.0
        if not has_trophy:
            # No trophy present at all — hard exclude for trophy queries
            return False
        if has_trophy and (high_imp or high_motion):
            # Confirmed high-importance trophy moment
            return True
        # Trophy present but not a highlight — fall through to token/description match
        return None

    # ── Drinking ──────────────────────────────────────────────────────────────
    if "drinking" in expanded_tokens and "bottle" not in (
        t for t in expanded_tokens if t != "drinking"
    ):
        # Only gate on action tag if the query is specifically about drinking
        if "drinking" in action_tags:
            return True
        # If drinking is in transcript treat it as a soft match — let scorer decide
        return None

    # ── Conversation  ─────────────────────────────────────────────────────────
    if "conversation" in expanded_tokens:
        if "conversation" in action_tags:
            return True
        return None

    # ── Running ───────────────────────────────────────────────────────────────
    if "running" in expanded_tokens:
        if "running" in action_tags:
            return True
        return None

    # ── Walking ───────────────────────────────────────────────────────────────
    if "walking" in expanded_tokens:
        if "walking" in action_tags:
            return True
        return None

    # ── Celebration / fast movement ───────────────────────────────────────────
    if "fast_movement" in expanded_tokens:
        if "fast_movement" in action_tags:
            return True
        # High motion alone is not sufficient — require an explicit tag so
        # driving scenes (also high-motion) don't get pulled in.
        return None

    return None   # no composite intent detected


# ── Main search function ───────────────────────────────────────────────────────

def search_events(
    query: str,
    scenes: list[SceneNarrative],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search the scene-level narrative timeline with a natural language query.

    Scoring per scene:
        transcript_score     * 0.40
      + dominant_score       * 0.20   (high-frequency / structurally important objects)
      + supporting_score     * 0.10   (present but not dominant — always included)
      + main_char_bonus      * 0.20   (1.0 if scene has any main character, else 0)
      + importance_score     * 0.10   (pre-normalised 0–1)

    Case-insensitive via _tokenise() lowercasing.

    Args:
        query:  Free-form user query string.
        scenes: Scene narratives from event_builder.build_events().
        top_k:  Maximum number of results to return.

    Returns:
        List of SearchResult sorted by score descending.
    """
    if not query.strip() or not scenes:
        return []

    # Basic token normalisation: strip trailing 's' for naive plural handling
    # (e.g. "cars" → "car", "plants" → "plant")
    query_tokens = {t.rstrip("s") for t in _tokenise(query)}

    # Semantic + intent expansion: add related terms from SEMANTIC_MAP and
    # INTENT_MAP so single-word queries surface semantically related objects,
    # action tags and composite scene intents without any ML model.
    expanded_tokens: set[str] = set(query_tokens)
    for token in query_tokens:
        if token in SEMANTIC_MAP:
            expanded_tokens.update(SEMANTIC_MAP[token])
        if token in INTENT_MAP:
            expanded_tokens.update(INTENT_MAP[token])

    results: list[SearchResult] = []

    for scene in scenes:
        # ── Strict token filter ───────────────────────────────────────────────
        # Require at least one query token to appear in the scene's transcript,
        # dominant objects, supporting objects, or action_tags before scoring.
        # This prevents importance-score noise from surfacing irrelevant scenes.
        transcript_text_lc = scene.transcript_summary.lower()
        dominant_lc   = [o.lower() for o in scene.dominant_objects]
        supporting_lc = [o.lower() for o in scene.supporting_objects]
        # Tokenise action tags (split on underscores so "person_with_ball" →
        # {"person", "with", "ball"} and each word is independently matchable).
        # Also keep the raw list for direct substring matching.
        action_tags = getattr(scene, "action_tags", [])
        action_tag_words = set()
        for tag in action_tags:
            action_tag_words.update(tag.lower().replace("_", " ").split())

        # ── Scene description (semantic fallback layer) ─────────────────────
        # Auto-generated natural-language summary; allows description-level
        # token matching as an additive layer on top of strict field matching.
        description = getattr(scene, "scene_description", "").lower()
        description_hit = any(token in description for token in expanded_tokens)

        match_found = False
        for token in expanded_tokens:
            if (
                token in transcript_text_lc
                or any(token in obj for obj in dominant_lc)
                or any(token in obj for obj in supporting_lc)
                or token in action_tag_words
                or any(token in tag.lower() for tag in action_tags)
            ):
                match_found = True
                break

        # ── Composite intent gate ─────────────────────────────────────────────
        # Evaluated BEFORE the per-token gate so composite logic can override
        # (either guarantee inclusion or hard-exclude) without re-checking tokens.
        composite_match = scene_matches_composite_intent(scene, expanded_tokens)

        if composite_match is False:
            continue

        if composite_match is None and not (match_found or description_hit):
            continue

        # ── Strict per-token object-presence gate ────────────────────────────
        # Skip entirely when the composite intent has already confirmed the
        # scene — the composite logic is authoritative in that case.
        # Otherwise: every original query token must appear somewhere in the
        # scene (transcript, objects, or action tags), OR one of its
        # semantic/intent expansions must match, OR the token is a known
        # filler word (common words that modify intent but carry no direct
        # scene signal: numbers, neutral qualifiers, etc.).
        all_objects_lc = dominant_lc + supporting_lc

        if composite_match is not True:
            for token in query_tokens:
                # Filler / neutral tokens — no direct scene signal required
                if token in _FILLER_TOKENS:
                    continue
                # Tokens whose INTENT_MAP entry is intentionally empty (e.g.
                # "moment", "scene") are treated as neutral qualifiers.
                if token in INTENT_MAP and not INTENT_MAP[token]:
                    continue

                in_transcript = token in transcript_text_lc
                in_objects    = any(token in obj for obj in all_objects_lc)
                in_actions    = (
                    token in action_tag_words
                    or any(token in t.lower() for t in action_tags)
                )

                if in_transcript or in_objects or in_actions:
                    continue   # raw token matched — OK

                # Try semantic + intent expansion for this token
                expansions: set[str] = set()
                if token in SEMANTIC_MAP:
                    expansions.update(SEMANTIC_MAP[token])
                if token in INTENT_MAP:
                    expansions.update(INTENT_MAP[token])

                if expansions:
                    exp_ok = any(
                        any(exp in obj for obj in all_objects_lc)
                        or exp in transcript_text_lc
                        or exp in action_tag_words
                        or any(exp in t.lower() for t in action_tags)
                        for exp in expansions
                    )
                    if exp_ok:
                        continue   # expansion matched — OK

                # No match for this token at all → exclude scene
                match_found = False
                break

            if not match_found:
                continue

        # ── Transcript match ──────────────────────────────────────────────────
        transcript_score = _overlap_score(query_tokens, scene.transcript_summary)

        # ── Dominant object match (weight 0.20) ───────────────────────────────
        # Per-object scoring: "person" earns only a minimal boost (prevents it
        # from inflating rank on every query via cached data where person may
        # still appear in dominant_objects).  All other objects get a strong
        # boost so rare objects rank above noisy person-heavy scenes.
        dominant_text  = " ".join(scene.dominant_objects)
        dominant_score = 0.0
        if scene.dominant_objects and query_tokens:
            dom_lc = [o.lower() for o in scene.dominant_objects]
            for token in query_tokens:
                for obj in dom_lc:
                    if token in obj:
                        dominant_score += 0.05 if obj == "person" else 0.35
                        break  # each query token contributes once
            dominant_score = min(dominant_score, 1.0)

        # ── Supporting object match (weight 0.10) ─────────────────────────────
        # Narrative dominance ≠ search relevance: always search supporting objs
        supporting_text  = " ".join(scene.supporting_objects)
        supporting_score = _overlap_score(query_tokens, supporting_text)

        # ── Main character bonus ──────────────────────────────────────────────
        main_char_bonus = 1.0 if scene.main_characters else 0.0

        # ── Importance score (pre-normalised) ─────────────────────────────────
        imp = scene.importance_score  # already in [0, 1]

        # ── Action tag score ──────────────────────────────────────────────────
        # Awards +0.4 per query token found in action tags (movement labels or
        # person–object interaction pairs).  Reuses action_tag_words computed
        # above for the strict filter — no duplicate computation.
        action_score = 0.0
        for token in expanded_tokens:
            if token in action_tag_words or any(token in tag.lower() for tag in action_tags):
                action_score += 0.4
        # Cap at 1.2 so a very long query can't produce absurdly high values
        action_score = min(action_score, 1.2)

        # ── Description match score ──────────────────────────────────────────
        # Awards +0.25 per expanded token found in the auto-generated description.
        # Capped at 1.0 so it supplements but does not overpower action / object
        # scoring (action_score cap is 1.2; description stays at most 1.0).
        description_score = 0.0
        for token in expanded_tokens:
            if token in description:
                description_score += 0.25
        description_score = min(description_score, 1.0)

        total_score = (
            TRANSCRIPT_WEIGHT        * transcript_score
            + DOMINANT_OBJECT_WEIGHT   * dominant_score
            + SUPPORTING_OBJECT_WEIGHT * supporting_score
            + MAIN_CHAR_WEIGHT         * main_char_bonus
            + IMPORTANCE_WEIGHT        * imp
            + action_score
            + description_score
        )

        if total_score < SCORE_THRESHOLD:
            continue

        matched = list(
            query_tokens & (
                _tokenise(scene.transcript_summary)
                | _tokenise(dominant_text)
                | _tokenise(supporting_text)
            )
        )

        # ── Explanation builder ───────────────────────────────────────────────
        # Summarise why this scene was surfaced so the UI can display reasoning.
        all_objects_for_exp = [
            o.lower() for o in scene.dominant_objects + scene.supporting_objects
        ]
        scene_action_tags = [t.lower() for t in getattr(scene, "action_tags", [])]
        transcript_lc     = scene.transcript_summary.lower()

        explanation_parts: list[str] = []
        seen_exp: set[str] = set()
        for token in sorted(expanded_tokens):
            key = token
            if key in seen_exp:
                continue
            if any(token in obj for obj in all_objects_for_exp):
                explanation_parts.append(f"object:{token}")
                seen_exp.add(key)
            elif token in scene_action_tags or any(token in tag for tag in scene_action_tags):
                explanation_parts.append(f"action:{token}")
                seen_exp.add(key)
            elif token in transcript_lc:
                explanation_parts.append(f"text:{token}")
                seen_exp.add(key)

        if composite_match is True:
            explanation_parts.insert(0, "composite:intent")

        if description_hit:
            explanation_parts.append("matched scene description")

        t = int(scene.start_time)
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        fmt = f"{h:02d}h{m:02d}m{s:02d}s"

        results.append(SearchResult(
            start_time=scene.start_time,
            end_time=scene.end_time,
            scene_id=scene.scene_id,
            score=total_score,
            dominant_objects=list(scene.dominant_objects),
            supporting_objects=list(scene.supporting_objects),
            main_characters=list(scene.main_characters),
            new_character_entry=list(scene.new_character_entry),
            importance_score=scene.importance_score,
            event_type=scene.event_type,
            transcript_text=scene.transcript_summary,
            formatted_time=fmt,
            matched_terms=matched,
            explanation=", ".join(explanation_parts),
            scene_description=getattr(scene, "scene_description", ""),
            matched_description=getattr(scene, "scene_description", "") if description_hit else "",
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def format_results_for_api(results: list[SearchResult]) -> dict:
    """
    Format search results for the Flask JSON response.
    """
    output = {}
    for i, result in enumerate(results):
        output[str(i)] = result.to_dict()
    return output
