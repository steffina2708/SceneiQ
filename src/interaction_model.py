"""
src/interaction_model.py
Computes motion intensity, scene type, person–object interactions, and movement
action tags from a group of FrameDetections belonging to a single scene.

Extracted from the scene synthesis pipeline for clean separation of concerns.
No ML models — pure spatial and temporal heuristics on normalised bounding boxes.
"""

import logging
import math
from collections import defaultdict

from .detection import FrameDetections

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
MOTION_ACTION_THRESHOLD = 1.2   # avg track speed above this → "action" scene
INTERACTION_PADDING = 0.12      # expand person bbox by 12% on each side for proximity


def compute_interactions(frames: list[FrameDetections]) -> dict:
    """
    Compute motion and interaction features for a set of frames from one scene.

    Analyses:
      A. Track positions and velocities (normalised coords / second)
      B. Class-specific movement tags (running, walking, vehicle_moving, ball_fast_motion)
      C. Scene motion intensity and type (action vs. static)
      D. Person–object proximity interactions
      E. Drinking detection (bottle in upper 40% of person bbox, ≥ 2 frames)
      F. Conversation detection (≥ 2 slow persons with centres < 0.3 apart)

    Args:
        frames: List of FrameDetections for all frames in this scene.

    Returns:
        dict with keys:
          - motion_intensity (float): average track speed across scene
          - scene_type (str): "action" or "static"
          - interactions (list[str]): person–object interaction labels
          - action_tags (list[str]): movement + interaction labels combined
    """
    # ── A. Build track positions (all classes, keyed by track_id) ─────────────
    track_positions: dict[int, list] = defaultdict(list)
    for frame in frames:
        for det in frame.detections:
            if det.track_id is not None:
                x1, y1, x2, y2 = det.box_xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                track_positions[det.track_id].append(
                    (frame.timestamp, det.class_name, cx, cy)
                )

    # ── B. Compute average velocity per track ─────────────────────────────────
    track_speeds: dict[int, float] = {}
    track_class: dict[int, str] = {}
    for track_id, positions in track_positions.items():
        if positions:
            track_class[track_id] = positions[0][1]
        speed_samples = []
        for i in range(1, len(positions)):
            t1, _, x1, y1 = positions[i - 1]
            t2, _, x2, y2 = positions[i]
            dt = max(t2 - t1, 1e-6)
            dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            speed_samples.append(dist / dt)
        if speed_samples:
            track_speeds[track_id] = sum(speed_samples) / len(speed_samples)

    # ── C. Class-specific movement classification ─────────────────────────────
    movement_tags: set[str] = set()
    for track_id, avg_speed in track_speeds.items():
        cls = track_class.get(track_id, "")
        if cls == "person":
            if avg_speed >= 1.8:
                movement_tags.add("running")
            elif avg_speed >= 0.6:
                movement_tags.add("walking")
        elif cls in ("car", "bicycle"):
            if avg_speed >= 1.5:
                movement_tags.add("vehicle_moving")
        elif cls == "sports ball":
            if avg_speed >= 2.0:
                movement_tags.add("ball_fast_motion")

    # ── D. Scene motion intensity ──────────────────────────────────────────────
    scene_motion = (
        sum(track_speeds.values()) / max(len(track_speeds), 1)
        if track_speeds else 0.0
    )
    motion_intensity = round(scene_motion, 3)
    scene_type = "action" if scene_motion > MOTION_ACTION_THRESHOLD else "static"

    # ── E. Person–object interactions (proximity, not strict overlap) ──────────
    interaction_tags: set[str] = set()
    for frame in frames:
        persons = [d for d in frame.detections if d.class_name == "person"]
        objects = [d for d in frame.detections if d.class_name != "person"]
        if not persons or not objects:
            continue
        for p in persons:
            px1, py1, px2, py2 = p.box_xyxy
            pad_x = (px2 - px1) * INTERACTION_PADDING
            pad_y = (py2 - py1) * INTERACTION_PADDING
            epx1 = max(0.0, px1 - pad_x)
            epy1 = max(0.0, py1 - pad_y)
            epx2 = min(1.0, px2 + pad_x)
            epy2 = min(1.0, py2 + pad_y)
            for obj in objects:
                ox1, oy1, ox2, oy2 = obj.box_xyxy
                ocx = (ox1 + ox2) / 2
                ocy = (oy1 + oy2) / 2
                if epx1 <= ocx <= epx2 and epy1 <= ocy <= epy2:
                    interaction_tags.add(f"person_with_{obj.class_name}")

    # ── F. Drinking detection ──────────────────────────────────────────────────
    # Rule: bottle centre in upper 40% of person bbox, in >= 2 frames.
    drinking_frames = 0
    for frame in frames:
        persons = [d for d in frame.detections if d.class_name == "person"]
        bottles = [d for d in frame.detections if d.class_name == "bottle"]
        if not persons or not bottles:
            continue
        frame_triggered = False
        for p in persons:
            if frame_triggered:
                break
            px1, py1, px2, py2 = p.box_xyxy
            upper_y = py1 + 0.4 * (py2 - py1)
            for b in bottles:
                bx1, by1, bx2, by2 = b.box_xyxy
                bocx = (bx1 + bx2) / 2
                bocy = (by1 + by2) / 2
                if px1 <= bocx <= px2 and py1 <= bocy <= upper_y:
                    frame_triggered = True
                    break
        if frame_triggered:
            drinking_frames += 1
    if drinking_frames >= 2:
        interaction_tags.add("drinking")

    # ── G. Conversation detection ──────────────────────────────────────────────
    # Rule: >= 2 tracked persons, both speed < 0.5, centroid distance < 0.3.
    conversation_found = False
    for frame in frames:
        if conversation_found:
            break
        slow_persons = [
            d for d in frame.detections
            if d.class_name == "person"
            and d.track_id is not None
            and track_speeds.get(d.track_id, 0.0) < 0.5
        ]
        if len(slow_persons) < 2:
            continue
        for i in range(len(slow_persons)):
            for j in range(i + 1, len(slow_persons)):
                ax1, ay1, ax2, ay2 = slow_persons[i].box_xyxy
                bx1, by1, bx2, by2 = slow_persons[j].box_xyxy
                acx = (ax1 + ax2) / 2
                acy = (ay1 + ay2) / 2
                bcx = (bx1 + bx2) / 2
                bcy = (by1 + by2) / 2
                if math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2) < 0.3:
                    conversation_found = True
                    break
            if conversation_found:
                break
    if conversation_found:
        interaction_tags.add("conversation")

    return {
        "motion_intensity": motion_intensity,
        "scene_type": scene_type,
        "interactions": list(interaction_tags),
        "action_tags": list(movement_tags | interaction_tags),
    }
