"""
src/retrieval.py
Multimodal semantic search over the scene-level narrative timeline.

Scoring strategy (scene-level):
  – Transcript match           (weight 0.40)
  – Dominant objects match     (weight 0.20)
  – Supporting objects match   (weight 0.10)
  – Main character bonus       (weight 0.20)
  – Importance score           (weight 0.10)
  – Action tag score           (additive, capped at 1.2)
  – Description match score    (additive, capped at 1.0)

Results are deduplicated by scene_id so the frontend shows one card per scene.
"""

import logging
from dataclasses import dataclass, field

from .scene_synthesizer import SceneNarrative
from .semantic_index import (
    SEMANTIC_MAP,
    INTENT_MAP,
    _FILLER_TOKENS,
    _tokenise,
    _overlap_score,
)

logger = logging.getLogger(__name__)

# ── Scoring weights ────────────────────────────────────────────────────────────
TRANSCRIPT_WEIGHT         = 0.40
DOMINANT_OBJECT_WEIGHT    = 0.20
SUPPORTING_OBJECT_WEIGHT  = 0.10
MAIN_CHAR_WEIGHT          = 0.20
IMPORTANCE_WEIGHT         = 0.10
SCORE_THRESHOLD           = 0.06


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


# ── Composite intent matcher ──────────────────────────────────────────────────

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

    # ── Driving scene ─────────────────────────────────────────────────────────
    if "vehicle_moving" in expanded_tokens:
        has_person  = any("person" in o for o in objects_lc)
        has_vehicle = any(o in ["car", "truck", "bus", "vehicle"] for o in objects_lc)
        is_moving   = "vehicle_moving" in action_tags
        return has_person and has_vehicle and is_moving

    # ── Trophy / world-cup moment ─────────────────────────────────────────────
    if "trophy" in expanded_tokens:
        has_trophy  = any("trophy" in o for o in objects_lc)
        high_imp    = scene.importance_score > 0.6
        high_motion = motion > 1.0
        if not has_trophy:
            return False
        if has_trophy and (high_imp or high_motion):
            return True
        return None

    # ── Drinking ──────────────────────────────────────────────────────────────
    if "drinking" in expanded_tokens and "bottle" not in (
        t for t in expanded_tokens if t != "drinking"
    ):
        if "drinking" in action_tags:
            return True
        return None

    # ── Conversation ──────────────────────────────────────────────────────────
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
        return None

    return None


# ── Main search function ──────────────────────────────────────────────────────

def search_events(
    query: str,
    scenes: list[SceneNarrative],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search the scene-level narrative timeline with a natural language query.

    Args:
        query:  Free-form user query string.
        scenes: Scene narratives from scene_synthesizer.build_events().
        top_k:  Maximum number of results to return.

    Returns:
        List of SearchResult sorted by score descending.
    """
    if not query.strip() or not scenes:
        return []

    # Naive plural normalisation: strip trailing 's'
    query_tokens = {t.rstrip("s") for t in _tokenise(query)}

    # Semantic + intent expansion
    expanded_tokens: set[str] = set(query_tokens)
    for token in query_tokens:
        if token in SEMANTIC_MAP:
            expanded_tokens.update(SEMANTIC_MAP[token])
        if token in INTENT_MAP:
            expanded_tokens.update(INTENT_MAP[token])

    results: list[SearchResult] = []

    for scene in scenes:
        transcript_text_lc = scene.transcript_summary.lower()
        dominant_lc   = [o.lower() for o in scene.dominant_objects]
        supporting_lc = [o.lower() for o in scene.supporting_objects]
        action_tags   = getattr(scene, "action_tags", [])
        action_tag_words = set()
        for tag in action_tags:
            action_tag_words.update(tag.lower().replace("_", " ").split())

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

        # Composite intent gate
        composite_match = scene_matches_composite_intent(scene, expanded_tokens)

        if composite_match is False:
            continue

        if composite_match is None and not (match_found or description_hit):
            continue

        all_objects_lc = dominant_lc + supporting_lc

        if composite_match is not True:
            for token in query_tokens:
                if token in _FILLER_TOKENS:
                    continue
                if token in INTENT_MAP and not INTENT_MAP[token]:
                    continue

                in_transcript = token in transcript_text_lc
                in_objects    = any(token in obj for obj in all_objects_lc)
                in_actions    = (
                    token in action_tag_words
                    or any(token in t.lower() for t in action_tags)
                )

                if in_transcript or in_objects or in_actions:
                    continue

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
                        continue

                match_found = False
                break

            if not match_found:
                continue

        # ── Transcript match ───────────────────────────────────────────────────
        transcript_score = _overlap_score(query_tokens, scene.transcript_summary)

        # ── Dominant object match ──────────────────────────────────────────────
        dominant_score = 0.0
        if scene.dominant_objects and query_tokens:
            dom_lc = [o.lower() for o in scene.dominant_objects]
            for token in query_tokens:
                for obj in dom_lc:
                    if token in obj:
                        dominant_score += 0.05 if obj == "person" else 0.35
                        break
            dominant_score = min(dominant_score, 1.0)

        # ── Supporting object match ────────────────────────────────────────────
        supporting_text  = " ".join(scene.supporting_objects)
        supporting_score = _overlap_score(query_tokens, supporting_text)

        # ── Main character bonus ───────────────────────────────────────────────
        main_char_bonus = 1.0 if scene.main_characters else 0.0

        # ── Importance score ───────────────────────────────────────────────────
        imp = scene.importance_score

        # ── Action tag score ───────────────────────────────────────────────────
        action_score = 0.0
        for token in expanded_tokens:
            if token in action_tag_words or any(token in tag.lower() for tag in action_tags):
                action_score += 0.4
        action_score = min(action_score, 1.2)

        # ── Description match score ────────────────────────────────────────────
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

        dominant_text = " ".join(scene.dominant_objects)
        matched = list(
            query_tokens & (
                _tokenise(scene.transcript_summary)
                | _tokenise(dominant_text)
                | _tokenise(supporting_text)
            )
        )

        # ── Explanation builder ───────────────────────────────────────────────
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
    """Format search results for the Flask JSON response."""
    output = {}
    for i, result in enumerate(results):
        output[str(i)] = result.to_dict()
    return output
