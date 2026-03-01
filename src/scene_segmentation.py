"""
src/scene_segmentation.py
Structural scene detection using PySceneDetect ContentDetector (HSV-based).

Replaces the legacy histogram-correlation approach with PySceneDetect's
ContentDetector, which analyses HSV colour-space differences and handles
fast-motion, lighting changes, and gradual scene transitions far more robustly.

Public API:
    segment_scenes(frames, video_path, ...)  → list[Scene]
    assign_scene_to_timestamp(scenes, ts)    → int
    Scene                                    dataclass

Explicit failure policy:
    - video_path not provided          → RuntimeError (hard failure)
    - PySceneDetect not installed      → RuntimeError (hard failure)
    - PySceneDetect raises at runtime  → logged as ERROR, re-raised (hard failure)
    - NO silent fallback to histogram  under any condition
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Scene dataclass (public contract) ─────────────────────────────────────────

@dataclass
class Scene:
    scene_id: int
    start_time: float
    end_time: float
    timestamps: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


# ── Motion level helper ───────────────────────────────────────────────────────

def _mean_motion_magnitude(frames: list[tuple[float, np.ndarray]]) -> float:
    """
    Compute average absolute pixel-difference magnitude across sampled frame pairs.
    Used to decide the PySceneDetect threshold (lower = more sensitive).
    """
    import cv2
    sample_step = max(1, len(frames) // 30)   # sample at most 30 pairs
    diffs = []
    for i in range(sample_step, len(frames), sample_step):
        prev_gray = cv2.cvtColor(frames[i - sample_step][1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i][1], cv2.COLOR_BGR2GRAY)
        diffs.append(float(np.mean(np.abs(curr_gray.astype(np.int16) - prev_gray.astype(np.int16)))))
    return float(np.mean(diffs)) if diffs else 0.0


# ── PySceneDetect integration ─────────────────────────────────────────────────

def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> list[dict]:
    """
    Run PySceneDetect ContentDetector on *video_path*.

    Supports both the modern API (scenedetect >= 0.6, open_video) and the
    legacy API (scenedetect < 0.6, VideoManager).

    Args:
        video_path: Absolute path to the video file.
        threshold:  ContentDetector HSV-weighted threshold.

    Returns:
        List of {"scene_id": int, "start": float, "end": float} dicts.

    Raises:
        RuntimeError: If PySceneDetect is not installed or raises during detection.
    """
    # ── Attempt modern API (scenedetect >= 0.6) ───────────────────────────────
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video       = open_video(video_path)
        scene_mgr   = SceneManager()
        scene_mgr.add_detector(ContentDetector(threshold=threshold))
        scene_mgr.detect_scenes(video)
        scene_list  = scene_mgr.get_scene_list()

        logger.info(
            f"[scene_segmentation] PySceneDetect (new API, threshold={threshold}) "
            f"→ {len(scene_list)} scenes"
        )
        return [
            {
                "scene_id": i,
                "start":    start.get_seconds(),
                "end":      end.get_seconds(),
            }
            for i, (start, end) in enumerate(scene_list)
        ]

    except ImportError:
        pass

    except AttributeError:
        pass

    except Exception as exc:
        logger.error(
            f"[scene_segmentation] PySceneDetect (new API) raised during detection: {exc}"
        )
        raise RuntimeError(
            f"Scene detection failed. PySceneDetect (new API) error: {exc}"
        ) from exc

    # ── Attempt legacy API (scenedetect < 0.6) ────────────────────────────────
    try:
        from scenedetect import VideoManager, SceneManager          # type: ignore
        from scenedetect.detectors import ContentDetector             # type: ignore

        video_mgr = VideoManager([video_path])
        scene_mgr = SceneManager()
        scene_mgr.add_detector(ContentDetector(threshold=threshold))
        video_mgr.set_downscale_factor()
        video_mgr.start()
        scene_mgr.detect_scenes(frame_source=video_mgr)
        scene_list = scene_mgr.get_scene_list(video_mgr.get_base_timecode())
        video_mgr.release()

        logger.info(
            f"[scene_segmentation] PySceneDetect (legacy API, threshold={threshold}) "
            f"→ {len(scene_list)} scenes"
        )
        return [
            {
                "scene_id": i,
                "start":    start.get_seconds(),
                "end":      end.get_seconds(),
            }
            for i, (start, end) in enumerate(scene_list)
        ]

    except ImportError as exc:
        logger.error(
            "[scene_segmentation] PySceneDetect is not installed. "
            "Install with: pip install scenedetect[opencv]"
        )
        raise RuntimeError(
            "Scene detection failed. PySceneDetect unavailable. "
            "Install with: pip install scenedetect[opencv]"
        ) from exc

    except Exception as exc:
        logger.error(
            f"[scene_segmentation] PySceneDetect (legacy API) raised during detection: {exc}"
        )
        raise RuntimeError(
            f"Scene detection failed. PySceneDetect (legacy API) error: {exc}"
        ) from exc


# ── Main public function ──────────────────────────────────────────────────────

def segment_scenes(
    frames: list[tuple[float, np.ndarray]],
    video_path: Optional[str] = None,
    threshold: float = 0.45,          # retained for call-site compatibility; ignored
    min_scene_duration: float = 3.0,  # retained for call-site compatibility; ignored
) -> list[Scene]:
    """
    Detect scene boundaries and group frames into Scene objects.

    Requires PySceneDetect and a valid video_path. Will not silently degrade.

    Args:
        frames:             List of (timestamp, BGR frame) tuples.
        video_path:         Path to the source video — REQUIRED.
        threshold:          Retained for call-site compatibility; ignored.
        min_scene_duration: Retained for call-site compatibility; ignored.

    Returns:
        List of Scene objects with scene_id, start_time, end_time, timestamps.

    Raises:
        RuntimeError: If video_path is not provided or detection fails.
    """
    if not frames:
        return []

    if not video_path:
        logger.error("[scene_segmentation] video_path is required but was not provided.")
        raise RuntimeError(
            "Scene detection failed. video_path is required for PySceneDetect."
        )

    # Auto-tune threshold: high-motion → 20, normal → 27
    motion = _mean_motion_magnitude(frames)
    psd_threshold = 20.0 if motion > 15.0 else 27.0
    logger.info(
        f"[scene_segmentation] Motion magnitude={motion:.2f} → "
        f"ContentDetector threshold={psd_threshold}"
    )

    scene_dicts = detect_scenes(video_path, threshold=psd_threshold)

    # Edge case: no cuts detected → entire video is one scene
    if not scene_dicts:
        logger.warning(
            "[scene_segmentation] No cuts detected — treating entire video as one scene."
        )
        all_ts = [t for t, _ in frames]
        return [Scene(
            scene_id=0,
            start_time=all_ts[0],
            end_time=all_ts[-1],
            timestamps=all_ts,
        )]

    # Build Scene objects, assigning frame timestamps
    video_end = frames[-1][0]
    scenes: list[Scene] = []
    for seg in scene_dicts:
        s_start = seg["start"]
        s_end   = seg["end"] if seg["end"] > 0 else video_end
        ts_list = [t for t, _ in frames if s_start <= t <= s_end]
        scenes.append(Scene(
            scene_id=seg["scene_id"],
            start_time=s_start,
            end_time=s_end,
            timestamps=ts_list,
        ))

    # Assign unclaimed timestamps (boundary rounding) to nearest scene
    claimed = {t for sc in scenes for t in sc.timestamps}
    unclaimed = [t for t, _ in frames if t not in claimed]
    for uts in unclaimed:
        best = min(scenes, key=lambda sc: min(abs(uts - sc.start_time), abs(uts - sc.end_time)))
        best.timestamps.append(uts)
        best.timestamps.sort()

    logger.info(f"[scene_segmentation] Built {len(scenes)} scenes via PySceneDetect.")
    return scenes


# ── Legacy histogram segmentation (manual use only — never auto-called) ───────

def legacy_histogram_segmentation(
    frames: list[tuple[float, np.ndarray]],
    threshold: float = 0.45,
    min_scene_duration: float = 3.0,
) -> list[Scene]:
    """
    Histogram-correlation scene segmentation.

    WARNING: This function is NOT called automatically under any condition.
    It exists solely for offline benchmarking or research comparison.
    """
    from .frame_sampler import compute_histogram

    scenes: list[Scene] = []
    scene_id             = 0
    current_scene_start  = frames[0][0]
    current_scene_ts     = [frames[0][0]]
    prev_hist            = compute_histogram(frames[0][1])

    for timestamp, frame in frames[1:]:
        curr_hist = compute_histogram(frame)
        correlation = float(
            np.corrcoef(prev_hist, curr_hist)[0, 1]
            if prev_hist.shape == curr_hist.shape
            else 0.0
        )
        if correlation < threshold and (timestamp - current_scene_start) >= min_scene_duration:
            scenes.append(Scene(
                scene_id=scene_id,
                start_time=current_scene_start,
                end_time=timestamp,
                timestamps=current_scene_ts.copy(),
            ))
            scene_id += 1
            current_scene_start = timestamp
            current_scene_ts    = [timestamp]
        else:
            current_scene_ts.append(timestamp)
        prev_hist = curr_hist

    if current_scene_ts:
        scenes.append(Scene(
            scene_id=scene_id,
            start_time=current_scene_start,
            end_time=frames[-1][0],
            timestamps=current_scene_ts,
        ))

    logger.info(f"[scene_segmentation] legacy_histogram_segmentation → {len(scenes)} scenes.")
    return scenes


# ── Timestamp → scene lookup ──────────────────────────────────────────────────

def assign_scene_to_timestamp(scenes: list[Scene], timestamp: float) -> int:
    """
    Return the scene_id that contains the given timestamp.

    Args:
        scenes:    List of Scene objects.
        timestamp: Timestamp in seconds.

    Returns:
        scene_id integer, or 0 if not found (defaults to first scene).
    """
    for scene in scenes:
        if scene.start_time <= timestamp <= scene.end_time:
            return scene.scene_id
    if scenes:
        nearest = min(scenes, key=lambda sc: min(abs(timestamp - sc.start_time), abs(timestamp - sc.end_time)))
        return nearest.scene_id
    return -1
