"""
src/pipeline.py
Orchestrates the full video processing pipeline:
  Download → Frame Extraction → Object Detection →
  Action Inference → Scene Segmentation → Transcript Fetch →
  Event Building → Cache

Results are cached to disk so re-runs are instant.
All paths are resolved relative to the project root (one level above src/).
"""

import json
import logging
import os
from urllib.parse import urlparse, parse_qs

from .transcript_engine import get_transcript as _get_transcript_hybrid
from .downloader import VideoDownloader
from .frame_sampler import frames_to_list, frames_to_list_adaptive
from .detection import ObjectDetector
from .motion_model import infer_action
from .scene_segmentation import segment_scenes
from .scene_synthesizer import build_events, events_to_json_timeline

logger = logging.getLogger(__name__)

# Resolve paths relative to project root (two levels up from this file in src/)
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DIR = os.path.join(_ROOT_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Singleton detector — loaded lazily on first use (Ultralytics auto-downloads model)
detector = ObjectDetector(confidence_threshold=0.35)


# ── URL utilities ──────────────────────────────────────────────────────────────

def get_video_id(url: str) -> str:
    """Extract YouTube video ID from any YouTube URL format."""
    if not url:
        return ""
    if "embed" in url:
        return url.split("/")[-1].split("?")[0]
    parsed = urlparse(url)
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("?")[0]
    query = parse_qs(parsed.query)
    if "v" in query:
        return query["v"][0]
    return ""


def _cache_path(video_id: str) -> str:
    return os.path.join(PROCESSED_DIR, f"{video_id}_events.json")


def _load_cache(video_id: str) -> dict | None:
    path = _cache_path(video_id)
    if os.path.exists(path):
        logger.info(f"Loading cached events for {video_id}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cache(video_id: str, timeline: dict) -> None:
    path = _cache_path(video_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)
    logger.info(f"Saved event cache: {path}")


# ── Transcript fetch ───────────────────────────────────────────────────────────

def fetch_transcript(video_id_or_url: str) -> list[dict]:
    """
    Hybrid transcript acquisition.

    Accepts either a bare video ID or a full YouTube URL.
    Tries youtube-transcript-api first; falls back to Whisper ASR if captions
    are unavailable, auto-generated, or empty.

    Returns list of {start, end, text} dicts, or empty list on hard failure.
    """
    if video_id_or_url.startswith("http"):
        url = video_id_or_url
    else:
        url = f"https://www.youtube.com/watch?v={video_id_or_url}"

    try:
        return _get_transcript_hybrid(url)
    except Exception as e:
        logger.warning(f"Hybrid transcript engine failed: {e}")
        return []


# ── Main pipeline entry ────────────────────────────────────────────────────────

def process_video(youtube_url: str, force_reprocess: bool = False) -> dict:
    """
    Run the full processing pipeline for a YouTube video.

    Args:
        youtube_url: Any valid YouTube URL.
        force_reprocess: Ignore cache and reprocess from scratch.

    Returns:
        JSON-serialisable timeline dict:
        {
          "video_id": str,
          "total_scenes": int,
          "scenes": [ {scene}, ... ]
        }
    """
    video_id = get_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Cannot extract video ID from: {youtube_url}")

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_reprocess:
        cached = _load_cache(video_id)
        if cached:
            return cached

    logger.info(f"Starting pipeline for video: {video_id}")

    # ── Step 1: Download ─────────────────────────────────────────────────────
    logger.info("STEP 1 — Downloading video …")
    uploads_dir = os.path.join(_ROOT_DIR, "uploads")
    downloader = VideoDownloader(output_dir=uploads_dir)
    result = downloader.download(youtube_url)
    if result["status"] == "error":
        raise Exception(result["error"])
    video_path = result["file_path"]

    # ── Step 2: Extract frames ───────────────────────────────────────────────
    logger.info("STEP 2 — Extracting frames (adaptive motion-aware) …")
    frames = frames_to_list_adaptive(video_path, max_dimension=640)
    logger.info(f"  → {len(frames)} frames extracted (adaptive rate 0.5–2 fps, cap=180).")

    # ── Step 3: Object detection ─────────────────────────────────────────────
    logger.info(f"STEP 3 — Running YOLOv8 object detection on {len(frames)} frames …")
    frame_detections = detector.detect_batch(frames, batch_size=4)

    # ── Step 4: Scene segmentation ───────────────────────────────────────────
    logger.info("STEP 4 — Segmenting scenes (PySceneDetect) …")
    scenes = segment_scenes(frames, video_path=video_path, threshold=0.45, min_scene_duration=3.0)
    logger.info(f"  → {len(scenes)} scenes found.")

    # ── Step 5: Action inference ─────────────────────────────────────────────
    logger.info("STEP 5 — Inferring actions …")
    transcript = fetch_transcript(youtube_url)
    action_results = []
    for i, (fd, (_, curr_frame)) in enumerate(zip(frame_detections, frames)):
        prev_frame = frames[i - 1][1] if i > 0 else None
        transcript_snippet = _get_transcript_snippet(transcript, fd.timestamp)
        action_results.append(
            infer_action(fd, transcript_snippet, prev_frame, curr_frame)
        )

    # ── Step 6: Build events ─────────────────────────────────────────────────
    logger.info("STEP 6 — Building scene-level narrative timeline …")
    narrative = build_events(frame_detections, action_results, scenes, transcript)

    # ── Serialise ─────────────────────────────────────────────────────────────
    timeline = events_to_json_timeline(narrative)
    timeline["video_id"] = video_id

    _save_cache(video_id, timeline)
    logger.info(f"Pipeline complete. {narrative.total_scenes} scenes built.")
    return timeline


def _get_transcript_snippet(transcript: list[dict], timestamp: float) -> str:
    """Return transcript text within ±2 seconds of timestamp."""
    parts = []
    for snippet in transcript:
        s = snippet.get("start", 0)
        if abs(s - timestamp) <= 2.0:
            parts.append(snippet.get("text", ""))
    return " ".join(parts).strip()
