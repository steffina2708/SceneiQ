"""
app.py
SceneIQ — Multimodal Scene Intelligence Engine
Flask backend with endpoints for video processing and semantic event search.
"""

import logging
import os
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template

from pipeline import process_video, get_video_id, fetch_transcript
from search.engine import search_events, format_results_for_api
from video_engine.event_builder import SceneNarrative

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# ── Upload cleanup config ──────────────────────────────────────────────────────
UPLOAD_FOLDER = Path("uploads")
FILE_TTL_SECONDS = 300  # 5 minutes

UPLOAD_FOLDER.mkdir(exist_ok=True)


def cleanup_worker():
    """Background thread: delete files in uploads/ older than FILE_TTL_SECONDS."""
    while True:
        now = time.time()
        if UPLOAD_FOLDER.exists():
            for file in UPLOAD_FOLDER.iterdir():
                try:
                    if file.is_file():
                        file_age = now - file.stat().st_mtime
                        if file_age > FILE_TTL_SECONDS:
                            file.unlink()
                            print(f"[SceneIQ] Deleted old file: {file.name}")
                except Exception as e:
                    print(f"[SceneIQ] Cleanup error: {e}")
        time.sleep(60)  # check every 60 seconds


_cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
_cleanup_thread.start()

# ── In-memory store for processed scene narratives ───────────────────────────
# Key: video_id → value: list[SceneNarrative] (deserialised for search)
_event_store: dict[str, list[SceneNarrative]] = {}
_processing_status: dict[str, str] = {}   # video_id → "processing" | "done" | "error:<msg>"
_timeline_cache: dict[str, dict] = {}     # video_id → raw JSON timeline


# ── Helper ─────────────────────────────────────────────────────────────────────

def _timeline_to_scenes(timeline: dict) -> list[SceneNarrative]:
    """
    Reconstruct SceneNarrative objects from a JSON timeline dict.

    Handles two formats:
      NEW (post-upgrade): timeline has a "scenes" key → direct reconstruction.
      OLD (pre-upgrade):  timeline has an "events" key → compatibility shim that
                          groups events by scene_id and synthesises SceneNarrative
                          objects so old cached videos remain searchable without
                          requiring a full reprocess.
    """
    # ── NEW format ────────────────────────────────────────────────────────────
    if "scenes" in timeline:
        scenes = []
        for raw in timeline["scenes"]:
            sn = SceneNarrative(
                scene_id=raw.get("scene_id", 0),
                start_time=raw.get("start_time", 0.0),
                end_time=raw.get("end_time", 0.0),
                main_characters=raw.get("main_characters", []),
                new_character_entry=raw.get("new_character_entry", []),
                track_ids=raw.get("track_ids", []),
                dominant_objects=raw.get("dominant_objects", []),
                supporting_objects=raw.get("supporting_objects", []),
                object_counts=raw.get("object_counts", {}),
                importance_score=raw.get("importance_score", 0.0),
                event_type=raw.get("event_type", "scene"),
                transcript_summary=raw.get("transcript_summary", ""),
                motion_intensity=raw.get("motion_intensity", 0.0),
                scene_type=raw.get("scene_type", "static"),
                interactions=raw.get("interactions", []),
                action_tags=raw.get("action_tags", []),
                scene_description=raw.get("scene_description", ""),
            )
            scenes.append(sn)
        return scenes

    # ── OLD format compatibility shim ─────────────────────────────────────────
    from collections import Counter as _Counter, defaultdict as _defaultdict
    from video_engine.event_builder import _compute_dominant_objects

    logger.info("Old-format cache detected — applying scene-compat migration shim.")
    events = timeline.get("events", [])
    if not events:
        return []

    # Group events by scene_id
    scene_buckets: dict = _defaultdict(list)
    for ev in events:
        scene_buckets[ev.get("scene_id", 0)].append(ev)

    scenes = []
    for scene_id in sorted(scene_buckets.keys()):
        evs = scene_buckets[scene_id]
        timestamps  = [e["start_time"] for e in evs]
        start_time  = min(timestamps)
        end_time    = max(e.get("end_time", e["start_time"] + 1.0) for e in evs)

        # Aggregate object counts across all events in this scene
        obj_counts: _Counter = _Counter()
        for ev in evs:
            for obj in ev.get("detected_objects", []):
                obj_counts[obj] += 1

        dominant, supporting = _compute_dominant_objects(dict(obj_counts))

        # Collect transcript text
        transcript_parts = [e.get("transcript_text", "") for e in evs if e.get("transcript_text")]
        seen, deduped = set(), []
        for part in transcript_parts:
            if part not in seen:
                seen.add(part)
                deduped.append(part)
        transcript_summary = " ".join(deduped).strip()

        # Simple importance proxy: person-count + dominant-object count
        person_count = sum(e.get("person_count", 0) for e in evs)
        importance_raw = len(dominant) + min(person_count, 3)

        sn = SceneNarrative(
            scene_id=scene_id,
            start_time=start_time,
            end_time=end_time,
            main_characters=[],
            new_character_entry=[],
            track_ids=[],
            dominant_objects=dominant,
            supporting_objects=supporting,
            object_counts=dict(obj_counts),
            importance_score=min(1.0, round(importance_raw / 10.0, 4)),
            event_type="scene",
            transcript_summary=transcript_summary,
        )
        scenes.append(sn)

    # Normalise importance scores to 0–1
    if scenes:
        max_imp = max(s.importance_score for s in scenes)
        if max_imp > 0:
            for s in scenes:
                s.importance_score = round(s.importance_score / max_imp, 4)

    logger.info(f"Compat shim built {len(scenes)} scene narratives from old-format cache.")
    return scenes


def _run_pipeline(youtube_url: str, video_id: str, force: bool) -> None:
    """Background thread target for video processing."""
    try:
        _processing_status[video_id] = "processing"
        logger.info(f"Background processing started: {video_id}")
        timeline = process_video(youtube_url, force_reprocess=force)
        _timeline_cache[video_id] = timeline
        _event_store[video_id] = _timeline_to_scenes(timeline)
        _processing_status[video_id] = "done"
        logger.info(f"Background processing complete: {video_id}")
    except Exception as e:
        logger.exception(f"Pipeline error for {video_id}: {e}")
        _processing_status[video_id] = f"error:{str(e)}"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


# ── Legacy endpoint (backward compat) ─────────────────────────────────────────

@app.route("/search_keyword", methods=["POST"])
def search_keyword():
    """
    Original keyword search endpoint.
    Falls back to transcript-only if video not yet processed,
    otherwise uses full multimodal search.
    """
    url = request.form.get("url", "")
    keyword = request.form.get("keyword", "")

    if not url or not keyword:
        return jsonify({}), 400

    video_id = get_video_id(url)

    # Multimodal path if events are available
    if video_id and video_id in _event_store:
        results = search_events(keyword, _event_store[video_id], top_k=20)
        output = {}
        for i, r in enumerate(results):
            fmt = r.formatted_time
            output[str(i)] = {fmt: r.start_time}
        return jsonify(output)

    # Fallback: transcript-only (original behaviour)
    if video_id:
        transcript = fetch_transcript(video_id)
        timestamps = []
        for snippet in transcript:
            if keyword.lower() in snippet.get("text", "").lower():
                timestamps.append(snippet["start"])

        if not timestamps:
            return jsonify({})

        result_dict = {}
        for i, t in enumerate(timestamps):
            m, s = divmod(t, 60)
            h, m = divmod(m, 60)
            fmt = "%dh%02dm%02ds" % (h, m, s)
            result_dict[str(i)] = {fmt: t}
        return jsonify(result_dict)

    return jsonify({}), 400


# ── New: Process video ─────────────────────────────────────────────────────────

@app.route("/process_video", methods=["POST"])
def process_video_endpoint():
    """
    Trigger full multimodal processing for a YouTube video.

    Body (JSON or form):
      url     — YouTube URL (required)
      force   — "true" to force reprocess (optional)

    Returns immediately with job status; poll /process_status/<video_id>.
    """
    data = request.get_json(silent=True) or request.form
    url = data.get("url", "")
    force = str(data.get("force", "false")).lower() == "true"

    if not url:
        return jsonify({"error": "url is required"}), 400

    video_id = get_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    status = _processing_status.get(video_id)

    if status == "processing":
        return jsonify({"video_id": video_id, "status": "processing"}), 202

    if status == "done" and not force:
        return jsonify({"video_id": video_id, "status": "done"}), 200

    # Start background processing
    thread = threading.Thread(
        target=_run_pipeline,
        args=(url, video_id, force),
        daemon=True,
    )
    thread.start()

    return jsonify({"video_id": video_id, "status": "processing"}), 202


@app.route("/process_status/<video_id>", methods=["GET"])
def process_status(video_id: str):
    """Poll processing status for a video."""
    status = _processing_status.get(video_id, "not_started")
    response = {"video_id": video_id, "status": status}

    if status == "done" and video_id in _timeline_cache:
        timeline = _timeline_cache[video_id]
        response["total_scenes"] = timeline.get("total_scenes", 0)

    return jsonify(response)


# ── New: Search events ─────────────────────────────────────────────────────────

@app.route("/search_events", methods=["POST"])
def search_events_endpoint():
    """
    Multimodal semantic search over a processed video's event timeline.

    Body (JSON or form):
      url     — YouTube URL (required)
      query   — Natural language search query (required)
      top_k   — Max results to return (optional, default 10)

    Returns:
      {
        "query": str,
        "video_id": str,
        "total_results": int,
        "results": { "0": {event}, "1": {event}, ... }
      }
    """
    data = request.get_json(silent=True) or request.form
    url = data.get("url", "")
    query = data.get("query", "")
    top_k = int(data.get("top_k", 10))

    if not url or not query:
        return jsonify({"error": "url and query are required"}), 400

    video_id = get_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    if video_id not in _event_store:
        # Try loading from disk cache
        import json, os
        cache_path = os.path.join("processed", f"{video_id}_events.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                timeline = json.load(f)
            _timeline_cache[video_id] = timeline
            _event_store[video_id] = _timeline_to_scenes(timeline)
        else:
            return jsonify({
                "error": "Video not yet processed. POST /process_video first.",
                "video_id": video_id,
            }), 404

    results = search_events(query, _event_store[video_id], top_k=top_k)
    formatted = format_results_for_api(results)

    return jsonify({
        "query": query,
        "video_id": video_id,
        "total_results": len(results),
        "results": formatted,
    })


# ── New: Get full timeline ─────────────────────────────────────────────────────

@app.route("/timeline/<video_id>", methods=["GET"])
def get_timeline(video_id: str):
    """Return the full structured event timeline for a processed video."""
    if video_id in _timeline_cache:
        return jsonify(_timeline_cache[video_id])

    import json, os
    cache_path = os.path.join("processed", f"{video_id}_events.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    return jsonify({"error": "Timeline not found. Process the video first."}), 404


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)
