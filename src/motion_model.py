"""
src/motion_model.py
Infers human actions from YOLO detections using deterministic heuristic rules.
No GPU, no model weights — pure logic derived from detected objects and spatial reasoning.

Rules are prioritised top-to-bottom; the first match wins.
"""

import logging
from dataclasses import dataclass

import numpy as np

from .detection import FrameDetections

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    action: str
    confidence: float     # 0.0 – 1.0 (rule-based estimate, not ML probability)
    reasoning: str        # Human-readable explanation for this inference


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _box_centre(box: list[float]) -> tuple[float, float]:
    """Return normalised centre (cx, cy) of a bounding box [x1,y1,x2,y2]."""
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _distance(c1: tuple, c2: tuple) -> float:
    return float(np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2))


def _persons_are_close(boxes: list[list[float]], threshold: float = 0.25) -> bool:
    """True if any two person boxes have centres within `threshold` (normalised)."""
    centres = [_box_centre(b) for b in boxes]
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            if _distance(centres[i], centres[j]) < threshold:
                return True
    return False


def _motion_magnitude(prev_frame: np.ndarray | None, curr_frame: np.ndarray | None) -> float:
    """
    Estimate inter-frame motion using mean absolute pixel difference.
    Returns value in [0, 255]; values > 20 indicate significant motion.
    """
    if prev_frame is None or curr_frame is None:
        return 0.0
    import cv2
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean())


# ── Rule engine ────────────────────────────────────────────────────────────────

def infer_action(
    frame_det: FrameDetections,
    transcript_snippet: str = "",
    prev_frame: np.ndarray | None = None,
    curr_frame: np.ndarray | None = None,
) -> ActionResult:
    """
    Infer a single primary action label for a video frame.

    Args:
        frame_det: YOLOv8 detection results for this frame.
        transcript_snippet: Any transcript text aligned to this timestamp.
        prev_frame: Previous frame (for motion estimation).
        curr_frame: Current frame (for motion estimation).

    Returns:
        ActionResult with action label, confidence, and reasoning.
    """
    objects = set(frame_det.object_names)
    person_boxes = frame_det.person_boxes
    has_person = "person" in objects
    person_count = len(person_boxes)
    motion = _motion_magnitude(prev_frame, curr_frame)

    # ── Rule 1: Presenting / Speaking ─────────────────────────────────────────
    if has_person and (
        "microphone" in objects
        or (transcript_snippet and len(transcript_snippet.strip()) > 10)
    ):
        return ActionResult(
            action="Speaking",
            confidence=0.85,
            reasoning="Person detected with microphone or concurrent transcript speech.",
        )

    # ── Rule 2: Sports activity ────────────────────────────────────────────────
    if has_person and "sports ball" in objects:
        return ActionResult(
            action="Playing sport",
            confidence=0.80,
            reasoning="Person and sports ball co-detected.",
        )

    # ── Rule 3: Typing / Working at computer ──────────────────────────────────
    if has_person and ("laptop" in objects or "keyboard" in objects or "mouse" in objects):
        return ActionResult(
            action="Working at computer",
            confidence=0.75,
            reasoning="Person near computing equipment.",
        )

    # ── Rule 4: Social interaction ─────────────────────────────────────────────
    if person_count >= 2 and _persons_are_close(person_boxes):
        return ActionResult(
            action="Social interaction",
            confidence=0.70,
            reasoning=f"{person_count} persons detected in close proximity.",
        )

    # ── Rule 5: Multiple people (crowd / group) ────────────────────────────────
    if person_count >= 3:
        return ActionResult(
            action="Group / Crowd scene",
            confidence=0.65,
            reasoning=f"{person_count} persons present.",
        )

    # ── Rule 6: Rapid motion / action scene ───────────────────────────────────
    if motion > 25.0 and has_person:
        return ActionResult(
            action="Active movement",
            confidence=0.60,
            reasoning=f"High inter-frame motion ({motion:.1f}) with person present.",
        )

    # ── Rule 7: Person on screen, static ──────────────────────────────────────
    if has_person:
        return ActionResult(
            action="Person on screen",
            confidence=0.50,
            reasoning="One or more persons detected, no specific action inferred.",
        )

    # ── Rule 8: No person ─────────────────────────────────────────────────────
    if objects:
        label = ", ".join(sorted(objects)[:3])
        return ActionResult(
            action="Scene: objects only",
            confidence=0.40,
            reasoning=f"No person detected. Visible objects: {label}.",
        )

    return ActionResult(
        action="Unknown / empty scene",
        confidence=0.20,
        reasoning="No significant detections.",
    )
