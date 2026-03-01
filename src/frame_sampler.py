"""
src/frame_sampler.py
Extracts frames from video using OpenCV.

Two strategies are available:
  • extract_frames / frames_to_list   – fixed-rate (legacy, default 1 fps)
  • extract_frames_adaptive / frames_to_list_adaptive
      Motion-aware adaptive sampling:
        high motion ( >HIGH_MOTION_THRESH ) → emit at most every 0.5 s
        normal motion ( >LOW_MOTION_THRESH ) → emit at most every 1.0 s
        static scene                          → emit at most every 2.0 s
      The video is probed at COARSE_HZ (2 Hz) using seek-based decoding,
      so no unnecessary frames are decoded.
"""

import math
import os
import logging
from typing import Generator, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Adaptive sampling constants ────────────────────────────────────────────────
COARSE_HZ: float = 2.0            # probe rate: seek every 0.5 s
HIGH_MOTION_THRESH: float = 15.0  # absdiff mean → emit every 0.5 s
LOW_MOTION_THRESH: float = 5.0    # absdiff mean → emit every 1.0 s
                                   # below LOW  → emit every 2.0 s

# ── Frame-count safety cap ────────────────────────────────────────────────────
MAX_FRAMES: int = 180  # hard ceiling; keeps CPU detection time bounded


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    max_dimension: int = 640,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Yield (timestamp_seconds, frame) tuples from a video file.

    Args:
        video_path: Path to the video file.
        fps: How many frames to sample per second (default 1).
        max_dimension: Resize the longer edge to this size for speed.

    Yields:
        (timestamp_in_seconds, BGR numpy array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(round(video_fps / fps))  # e.g. every 30 frames for 1fps

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            # Use codec-reported position for accuracy (avoids VFR / header drift)
            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)

            # Resize for speed
            h, w = frame.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            yield timestamp, frame

        frame_index += 1

    cap.release()
    logger.info(f"Extracted frames from: {video_path}")


def frames_to_list(
    video_path: str,
    fps: float = 1.0,
    max_dimension: int = 640,
) -> list[Tuple[float, np.ndarray]]:
    """
    Convenience wrapper — returns all (timestamp, frame) pairs as a list.
    Only use for short videos; for long videos, use extract_frames() generator.
    """
    return list(extract_frames(video_path, fps=fps, max_dimension=max_dimension))


def extract_frames_adaptive(
    video_path: str,
    max_dimension: int = 640,
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """
    Yield (timestamp_seconds, frame) tuples using motion-aware adaptive sampling.

    The video is seek-decoded at COARSE_HZ (2 fps). Between each probe pair the
    mean absolute pixel difference (grayscale) is computed and used to decide how
    soon the *next* frame may be emitted:

        motion > HIGH_MOTION_THRESH  ->  min_interval = 0.5 s  (up to 2 fps)
        motion > LOW_MOTION_THRESH   ->  min_interval = 1.0 s  (up to 1 fps)
        else                         ->  min_interval = 2.0 s  (up to 0.5 fps)

    Args:
        video_path:    Path to the video file.
        max_dimension: Resize the longer edge to this size before any processing.

    Yields:
        (timestamp_in_seconds, BGR numpy array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / video_fps if total_frames > 0 else 0.0

    # Seek step: one probe every COARSE_HZ seconds
    seek_interval = video_fps / COARSE_HZ  # frames between probes

    prev_gray = None
    last_emitted_ts: float = -999.0
    emitted = 0

    def _resize(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            return cv2.resize(frame, (int(w * scale), int(h * scale)))
        return frame

    probe_index = 0
    while True:
        target_frame = int(round(probe_index * seek_interval))
        if total_frames > 0 and target_frame >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            break

        # Read the actual decoded timestamp from the codec — this is more
        # accurate than target_frame / video_fps because H.264/H.265 seek
        # lands on the nearest keyframe, not the exact requested frame.
        timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
        frame = _resize(frame)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute motion score against previous probe
        if prev_gray is not None and prev_gray.shape == curr_gray.shape:
            motion = float(cv2.absdiff(prev_gray, curr_gray).mean())
        else:
            motion = HIGH_MOTION_THRESH + 1.0  # always emit the first frame

        # Choose minimum interval based on motion magnitude
        if motion > HIGH_MOTION_THRESH:
            min_interval = 0.5
        elif motion > LOW_MOTION_THRESH:
            min_interval = 1.0
        else:
            min_interval = 2.0

        if (timestamp - last_emitted_ts) >= min_interval:
            yield timestamp, frame
            last_emitted_ts = timestamp
            emitted += 1

        prev_gray = curr_gray
        probe_index += 1

    cap.release()
    logger.info(
        f"Adaptive extraction: {emitted} frames from {video_path} "
        f"(duration ~{duration:.0f}s, probe rate {COARSE_HZ} Hz)"
    )


def frames_to_list_adaptive(
    video_path: str,
    max_dimension: int = 640,
) -> list[Tuple[float, np.ndarray]]:
    """
    Convenience wrapper — returns all adaptive-sampled (timestamp, frame) pairs
    as a list. The sampling rate varies from 0.5 fps (static) to 2 fps (high
    motion) based on inter-frame pixel difference.

    If the first pass yields more than MAX_FRAMES frames the list is uniformly
    thinned so that at most MAX_FRAMES frames reach the detector.  This keeps
    CPU processing time bounded regardless of video length or motion level.
    """
    frames = list(extract_frames_adaptive(video_path, max_dimension=max_dimension))

    if len(frames) > MAX_FRAMES:
        step = math.ceil(len(frames) / MAX_FRAMES)
        frames = frames[::step]
        logger.info(
            f"MAX_FRAMES cap ({MAX_FRAMES}) applied: "
            f"kept {len(frames)} frames (uniform step={step})"
        )

    return frames


def compute_histogram(frame: np.ndarray) -> np.ndarray:
    """
    Compute a normalised HSV histogram for a frame.
    Used by the scene segmenter.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
