"""
src/scene_synthesizer.py
Combines YOLO detections, action inferences, scene boundaries, and transcript
data into a structured scene-level narrative event timeline.

Each SceneNarrative covers one structural scene and stores:
  - Dominant / supporting objects (aggregated across all frames in the scene)
  - Main characters (track_ids appearing in >= 3 scenes globally)
  - New character entries (track_ids first seen in this scene)
  - Transcript summary aligned to the scene window
  - Importance score (normalised 0–1)
  - Motion intelligence via interaction_model

The legacy Event dataclass is retained for any backward-compatible code paths.
"""

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

from .detection import FrameDetections
from .motion_model import ActionResult
from .scene_segmentation import Scene, assign_scene_to_timestamp
from .interaction_model import compute_interactions

logger = logging.getLogger(__name__)


# ── Legacy Event dataclass (retained for backward-compatible paths) ───────────

@dataclass
class Event:
    """A single timestamped multimodal event in the video. (legacy)"""

    start_time: float
    end_time: float
    scene_id: int
    detected_objects: list[str]
    person_count: int
    action: str
    action_confidence: float
    action_reasoning: str
    transcript_text: str
    event_id: int = 0
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def formatted_time(self) -> str:
        t = int(self.start_time)
        h, remainder = divmod(t, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"


# ── Scene-level narrative dataclasses ─────────────────────────────────────────

@dataclass
class SceneNarrative:
    """Structured narrative summary for a single scene."""

    scene_id: int
    start_time: float
    end_time: float

    # Character intelligence
    main_characters: list[int]        # track_ids appearing in >= 3 scenes globally
    new_character_entry: list[int]    # track_ids first seen in this scene
    track_ids: list[int]              # all track_ids observed in this scene

    # Object intelligence
    dominant_objects: list[str]       # high-frequency / high-coverage objects
    supporting_objects: list[str]     # low-frequency objects
    object_counts: dict[str, int]     # raw per-class detection count

    # Narrative scoring
    importance_score: float           # normalised 0–1
    event_type: str                   # "scene" | "character_introduction"

    # Language
    transcript_summary: str

    # Motion intelligence
    motion_intensity: float = 0.0
    scene_type: str = "static"           # "action" | "static"
    interactions: list[str] = field(default_factory=list)
    action_tags: list[str] = field(default_factory=list)
    scene_description: str = ""          # auto-generated natural language description

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def formatted_time(self) -> str:
        t = int(self.start_time)
        h, remainder = divmod(t, 3600)
        m, s = divmod(remainder, 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"


@dataclass
class NarrativeOutput:
    """Top-level container for the full scene-level narrative."""

    scenes: list[SceneNarrative]
    total_scenes: int
    main_characters_global: list[int]

    def to_dict(self) -> dict:
        return {
            "total_scenes": self.total_scenes,
            "main_characters_global": self.main_characters_global,
            "scenes": [s.to_dict() for s in self.scenes],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _align_transcript(
    transcript: list[dict],
    start_time: float,
    end_time: float,
) -> str:
    """
    Return concatenated transcript text that overlaps [start_time, end_time].

    Supports both:
        - New format : {"start", "end", "text"}
        - Legacy format: {"start", "duration", "text"}
    """
    parts = []
    for snippet in transcript:
        s = snippet.get("start", 0)
        if "end" in snippet:
            e = snippet["end"]
        else:
            e = s + snippet.get("duration", 1)
        if s <= end_time and e >= start_time:
            parts.append(snippet.get("text", "").strip())
    return " ".join(parts).strip()


def _compute_dominant_objects(
    object_counts: dict[str, int],
) -> tuple[list[str], list[str]]:
    """
    Split objects into dominant and supporting classes.

    Dominant criteria (either):
        - count >= 2                         (absolute minimum frequency)
        - count >= 30% of max scene count   AND count >= 2

    Objects that appear exactly once are ALWAYS supporting, never dominant.
    "person" is always routed to supporting; character intelligence is tracked
    separately via main_characters / new_character_entry (track_ids).

    Returns:
        (dominant_objects, supporting_objects) — both sorted alphabetically.
    """
    if not object_counts:
        return [], []

    non_person_counts = {obj: c for obj, c in object_counts.items() if obj != "person"}

    if non_person_counts:
        max_count = max(non_person_counts.values())
    else:
        max_count = max(object_counts.values())
    threshold = max(2, max_count * 0.30)

    dominant, supporting = [], []
    for obj, count in object_counts.items():
        if obj == "person":
            supporting.append(obj)
        elif count >= threshold:
            dominant.append(obj)
        else:
            supporting.append(obj)
    return sorted(dominant), sorted(supporting)


# ── Scene description generator ──────────────────────────────────────────────

def generate_scene_description(scene: dict) -> str:
    """
    Generate a deterministic natural-language description for a scene.

    Uses only fields already computed in the pipeline (scene_type, action_tags,
    dominant_objects, supporting_objects, transcript_summary) — no LLM required.

    Args:
        scene: A dict with SceneNarrative fields (e.g. from asdict(sn)).

    Returns:
        A plain English sentence-group describing the scene.
    """
    parts: list[str] = []

    if scene.get("scene_type") == "action":
        parts.append("This is a high-action scene.")
    else:
        parts.append("This is a dialogue or static scene.")

    if scene.get("action_tags"):
        parts.append(
            "Detected actions include "
            + ", ".join(scene["action_tags"]) + "."
        )

    if scene.get("dominant_objects"):
        parts.append(
            "Key objects visible are "
            + ", ".join(scene["dominant_objects"]) + "."
        )

    if scene.get("supporting_objects"):
        parts.append(
            "Other objects include "
            + ", ".join(scene["supporting_objects"]) + "."
        )

    if scene.get("transcript_summary"):
        parts.append("Spoken content: " + scene["transcript_summary"])

    return " ".join(parts)


# ── Main build function ───────────────────────────────────────────────────────

def build_events(
    frame_detections: list[FrameDetections],
    action_results: list[ActionResult],
    scenes: list[Scene],
    transcript: list[dict],
    window_size: float = 1.0,
) -> NarrativeOutput:
    """
    Construct a scene-level NarrativeOutput from per-frame analysis.

    Steps:
        1. Group frame detections into scenes
        2. Compute dominant / supporting objects per scene
        3. Detect new character (track_id) entries per scene
        4. Detect main characters (appear in >= 3 scenes globally)
        5. Compute normalised importance score per scene
        6. Build structured SceneNarrative objects

    Args:
        frame_detections: One FrameDetections per sampled frame.
        action_results:   One ActionResult per sampled frame (same order).
        scenes:           Scene segments from the scene segmenter.
        transcript:       Raw transcript list from the transcript engine.
        window_size:      Seconds used to extend the end_time of a lone frame.

    Returns:
        NarrativeOutput containing a SceneNarrative per scene.
    """
    if len(frame_detections) != len(action_results):
        raise ValueError("frame_detections and action_results must have equal length.")

    # ── STEP 1: Group detections by scene ─────────────────────────────────────
    scene_map: dict[int, dict] = {}
    for scene in scenes:
        scene_map[scene.scene_id] = {
            "scene_ref": scene,
            "frames": [],
            "track_ids": set(),
            "track_id_counts": Counter(),
            "object_counts": Counter(),
            "timestamps": [],
            "action_confidences": [],
        }

    for fd, ar in zip(frame_detections, action_results):
        scene_id = assign_scene_to_timestamp(scenes, fd.timestamp)
        if scene_id not in scene_map:
            scene_map[scene_id] = {
                "scene_ref": None,
                "frames": [],
                "track_ids": set(),
                "track_id_counts": Counter(),
                "object_counts": Counter(),
                "timestamps": [],
                "action_confidences": [],
            }
        bucket = scene_map[scene_id]
        bucket["frames"].append(fd)
        bucket["timestamps"].append(fd.timestamp)
        bucket["action_confidences"].append(ar.confidence)
        frame_track_ids: set[int] = set()
        for det in fd.detections:
            bucket["object_counts"][det.class_name] += 1
            if det.track_id is not None:
                bucket["track_ids"].add(det.track_id)
                frame_track_ids.add(det.track_id)
        for tid in frame_track_ids:
            bucket["track_id_counts"][tid] += 1

    # ── STEP 1b: Stability filter (remove single-frame noise) ─────────────────
    for bucket in scene_map.values():
        bucket["object_counts"] = Counter(
            {k: v for k, v in bucket["object_counts"].items() if v >= 2}
        )
        stable_track_ids = {
            tid for tid, cnt in bucket["track_id_counts"].items() if cnt >= 2
        }
        bucket["track_ids"] = bucket["track_ids"] & stable_track_ids

    # ── STEP 1c: Motion, velocity, interactions, action tags ──────────────────
    for scene_id, bucket in scene_map.items():
        interaction_data = compute_interactions(bucket["frames"])
        bucket["motion_intensity"] = interaction_data["motion_intensity"]
        bucket["scene_type"]       = interaction_data["scene_type"]
        bucket["interactions"]     = interaction_data["interactions"]
        bucket["action_tags"]      = interaction_data["action_tags"]
        logger.info(
            "Scene %s: %d track_ids, %d action_tags: %s",
            scene_id,
            len(bucket["track_ids"]),
            len(bucket["action_tags"]),
            bucket["action_tags"],
        )

    # ── STEP 2: Dominant / supporting objects ─────────────────────────────────
    for bucket in scene_map.values():
        dominant, supporting = _compute_dominant_objects(dict(bucket["object_counts"]))
        bucket["dominant_objects"] = dominant
        bucket["supporting_objects"] = supporting

    # ── STEP 3: Character entry detection ─────────────────────────────────────
    seen_track_ids: set[int] = set()
    for scene_id in sorted(scene_map.keys()):
        bucket = scene_map[scene_id]
        new_chars = bucket["track_ids"] - seen_track_ids
        bucket["new_character_entry"] = sorted(new_chars)
        seen_track_ids.update(bucket["track_ids"])
        bucket["event_type"] = "character_introduction" if new_chars else "scene"

    # ── STEP 4: Main character detection ──────────────────────────────────────
    track_scene_counts: Counter = Counter()
    for bucket in scene_map.values():
        for tid in bucket["track_ids"]:
            track_scene_counts[tid] += 1

    main_characters_global = sorted(
        tid for tid, cnt in track_scene_counts.items() if cnt >= 3
    )

    for bucket in scene_map.values():
        bucket["main_characters"] = sorted(
            t for t in bucket["track_ids"] if track_scene_counts[t] >= 3
        )

    # ── STEP 5: Scene importance score ────────────────────────────────────────
    raw_scores = []
    for bucket in scene_map.values():
        confs = bucket["action_confidences"]
        intensity = sum(confs) / len(confs) if confs else 0.0
        raw = (
            len(bucket["main_characters"]) * 2
            + len(bucket["new_character_entry"]) * 3
            + len(bucket["dominant_objects"])
            + intensity
        )
        bucket["raw_importance"] = raw
        raw_scores.append(raw)

    max_raw = max(raw_scores) if raw_scores else 1.0
    max_raw = max_raw if max_raw > 0 else 1.0

    for bucket in scene_map.values():
        bucket["importance_score"] = round(bucket["raw_importance"] / max_raw, 4)

    # ── STEP 6: Build SceneNarrative objects ──────────────────────────────────
    narrative_scenes: list[SceneNarrative] = []
    for scene_id in sorted(scene_map.keys()):
        bucket = scene_map[scene_id]
        timestamps = bucket["timestamps"]
        scene_ref: Optional[Scene] = bucket.get("scene_ref")

        if scene_ref is not None:
            start_time = scene_ref.start_time
            end_time = scene_ref.end_time
        elif timestamps:
            start_time = min(timestamps)
            end_time = round(max(timestamps) + window_size, 2)
        else:
            continue  # empty scene with no frames — skip

        transcript_summary = _align_transcript(transcript, start_time, end_time)

        sn = SceneNarrative(
            scene_id=scene_id,
            start_time=start_time,
            end_time=end_time,
            main_characters=bucket["main_characters"],
            new_character_entry=bucket["new_character_entry"],
            track_ids=sorted(bucket["track_ids"]),
            dominant_objects=bucket["dominant_objects"],
            supporting_objects=bucket["supporting_objects"],
            object_counts=dict(bucket["object_counts"]),
            importance_score=bucket["importance_score"],
            event_type=bucket["event_type"],
            transcript_summary=transcript_summary,
            motion_intensity=bucket.get("motion_intensity", 0.0),
            scene_type=bucket.get("scene_type", "static"),
            interactions=bucket.get("interactions", []),
            action_tags=bucket.get("action_tags", []),
        )
        sn.scene_description = generate_scene_description(sn.to_dict())
        narrative_scenes.append(sn)

    output = NarrativeOutput(
        scenes=narrative_scenes,
        total_scenes=len(narrative_scenes),
        main_characters_global=main_characters_global,
    )

    logger.info(
        "Built %d scene narratives | %d main characters globally.",
        len(narrative_scenes),
        len(main_characters_global),
    )
    return output


def events_to_json_timeline(narrative: NarrativeOutput) -> dict:
    """
    Serialise a NarrativeOutput into the canonical JSON structure.

    Returns:
        {
          "total_scenes": int,
          "main_characters_global": [...],
          "scenes": [ {...}, ... ]
        }
    """
    return narrative.to_dict()
