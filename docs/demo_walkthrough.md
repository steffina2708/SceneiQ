# SceneIQ — Demo Walkthrough

> A step-by-step guide to running SceneIQ end-to-end on any YouTube video.

---

## Prerequisites

- Python 3.10+
- `ffmpeg` on your system PATH (see [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html))
- All Python dependencies installed via `pip install -r requirements.txt`

---

## Step 1 — Start the Application

```bash
python run.py
```

Open your browser at [http://localhost:8080](http://localhost:8080).

---

## Step 2 — Load a YouTube Video

1. Paste any YouTube URL into the **URL field** (e.g. `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
2. Click **Load** — the video thumbnail will appear in the player.

---

## Step 3 — Analyse the Video

Click **Analyse**. SceneIQ will:

| Step | What happens |
|------|-------------|
| 1 | Download the video via `yt-dlp` |
| 2 | Extract frames (adaptive 0.5–2 fps based on motion) |
| 3 | Run YOLOv8s object detection + tracking on each frame |
| 4 | Segment scenes with PySceneDetect ContentDetector |
| 5 | Infer per-frame actions via rule-based motion heuristics |
| 6 | Fetch transcript (YouTube captions → Whisper ASR fallback) |
| 7 | Build scene-level narrative with importance scores |

Processing a typical 5-minute video takes **1–3 minutes** on a modern CPU.

A progress overlay shows the current pipeline step. Once complete, the **Scene Timeline** strip populates at the bottom of the left panel.

---

## Step 4 — Explore the Scene Timeline

Each **Scene chip** in the timeline shows:
- Scene number and start timestamp
- Scene type (Scene / character_introduction)
- ⭐ star if `importance_score ≥ 0.7`

Click any chip to **seek the video player** to that scene's start.

---

## Step 5 — Search

Type a natural-language query in the header search bar:

```
person near car
trophy moment
drinking water
two people talking
```

Hit **Search**. SceneIQ runs multimodal scoring across:
- Transcript text (weight 0.40)
- Dominant objects (weight 0.20)
- Supporting objects (weight 0.10)
- Character presence (weight 0.20)
- Importance score (weight 0.10)
- Action tags (additive, motion/interaction labels)
- Scene description (additive, NL summary layer)

Results appear in the right panel. Each card shows:
- **Timestamp** (click to seek)
- **Motion badge** (High Motion ⚡ or Static)
- **Relevance %** and **Importance %**
- **Scene type** label
- **Auto-generated description** (or matched text excerpt)
- **Object chips** — blue = dominant, yellow = supporting
- **"Why it matched"** explanation chips

---

## Step 6 — Re-analyse or Force Reprocess

Results are cached to `processed/<video_id>_events.json`. On second load, the cache is served instantly.

To force a fresh run, POST with `force=true`:

```bash
curl -X POST http://localhost:8080/process_video \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=...", "force": "true"}'
```

---

## REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/process_video` | Trigger async processing |
| `GET`  | `/process_status/<video_id>` | Poll job status |
| `POST` | `/search_events` | Multimodal semantic search |
| `GET`  | `/timeline/<video_id>` | Full structured timeline |

---

## Architecture Diagrams

> See [architecture.png](architecture.png) and [system_flow.png](system_flow.png) for visual overviews.
> *(Place diagram images in this directory.)*
