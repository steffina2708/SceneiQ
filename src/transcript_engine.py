"""
src/transcript_engine.py
Hybrid Transcript Acquisition Engine

Priority order:
  1. YouTube official / manual captions via youtube-transcript-api
  2. Auto-generated captions (accepted if non-empty)
  3. OpenAI Whisper local ASR fallback (for any video missing captions)

Output always conforms to:
    [
        {"start": float, "end": float, "text": str},
        ...
    ]

Usage:
    from src.transcript_engine import get_transcript
    segments = get_transcript("https://www.youtube.com/watch?v=VIDEO_ID")
"""

import json as _json
import logging
import math
import os
import subprocess
import wave
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "small"    # "tiny" | "small" | "medium" | "large"
CHUNK_MINUTES      = 10         # Chunk audio every N minutes for long videos

# ── Paths — resolved relative to project root (one level above src/) ──────────
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(_BASE_DIR, "uploads")
AUDIO_DIR   = os.path.join(UPLOADS_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)


# ── ffmpeg binary resolution ───────────────────────────────────────────────────
def _find_ffmpeg() -> str:
    """
    Return the ffmpeg command to use.

    Checks (in order):
    1. FFMPEG_PATH environment variable (user override).
    2. 'ffmpeg' on the system PATH (standard installs on Linux, macOS, Windows).

    Returns the command string (e.g. "ffmpeg" or a full absolute path).
    Raises RuntimeError if ffmpeg cannot be found.
    """
    # 1. Explicit override
    override = os.environ.get("FFMPEG_PATH")
    if override and Path(override).is_file():
        return override

    # 2. On PATH already?
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"

    raise RuntimeError(
        "ffmpeg binary not found.\n"
        "  • Linux   : sudo apt install ffmpeg\n"
        "  • macOS   : brew install ffmpeg\n"
        "  • Windows : Download from https://ffmpeg.org/download.html,\n"
        "              add its 'bin' folder to your PATH,\n"
        "              or set the FFMPEG_PATH environment variable."
    )


FFMPEG_CMD: str = ""   # resolved lazily on first use


def _ffmpeg() -> str:
    """Return the resolved ffmpeg command (lazy singleton)."""
    global FFMPEG_CMD
    if not FFMPEG_CMD:
        FFMPEG_CMD = _find_ffmpeg()
    return FFMPEG_CMD


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_video_id(url: str) -> str:
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


def _segments_from_api(transcript_iterable) -> list[dict]:
    """Convert youtube-transcript-api objects → standard segment dicts."""
    segments = []
    for s in transcript_iterable:
        text = s.text.strip() if hasattr(s, "text") else str(s.get("text", "")).strip()
        if not text:
            continue
        start    = s.start    if hasattr(s, "start")    else float(s.get("start", 0))
        duration = s.duration if hasattr(s, "duration") else float(s.get("duration", 1))
        segments.append({
            "start": round(start, 3),
            "end":   round(start + duration, 3),
            "text":  text,
        })
    return segments


# ── Audio extraction ──────────────────────────────────────────────────────────

def extract_audio(video_url: str) -> str:
    """
    Extract audio from a YouTube URL and convert to 16 kHz mono WAV.

    Reuses an existing WAV if already present.
    If the video was already downloaded as MP4/MKV/WEBM, extracts audio from
    that file to avoid a second network download.

    Returns:
        Absolute path to the 16 kHz mono WAV file.

    Raises:
        ValueError:  if a video ID cannot be parsed from the URL.
        FileNotFoundError: if audio extraction ultimately fails.
        ImportError: if yt_dlp is not installed.
    """
    try:
        from yt_dlp import YoutubeDL  # noqa: F401
    except ImportError:
        raise ImportError("yt_dlp is required. Install with: pip install yt-dlp")

    video_id = _get_video_id(video_url)
    if not video_id:
        raise ValueError(f"Cannot extract video ID from URL: {video_url}")

    wav_path = os.path.join(AUDIO_DIR, f"{video_id}.wav")

    # Already extracted?
    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
        logger.info(f"[transcript_engine] Reusing existing audio: {wav_path}")
        return wav_path

    # Reuse an already-downloaded video file
    for ext in ("mp4", "mkv", "webm"):
        candidate = os.path.join(UPLOADS_DIR, f"{video_id}.{ext}")
        if os.path.exists(candidate):
            logger.info(f"[transcript_engine] Extracting audio from existing video: {candidate}")
            _convert_to_wav(candidate, wav_path)
            return wav_path

    # Download audio only via yt_dlp
    logger.info(f"[transcript_engine] Downloading audio for {video_id} …")
    raw_prefix = os.path.join(AUDIO_DIR, f"{video_id}_raw")

    from yt_dlp import YoutubeDL
    ydl_opts = {
        "format":      "bestaudio/best",
        "outtmpl":     raw_prefix + ".%(ext)s",
        "quiet":       True,
        "noplaylist":  True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    raw_file = None
    for candidate in Path(AUDIO_DIR).glob(f"{video_id}_raw.*"):
        raw_file = str(candidate)
        break

    if raw_file is None:
        raise FileNotFoundError(
            f"[transcript_engine] yt_dlp did not produce an audio file for {video_id}"
        )

    _convert_to_wav(raw_file, wav_path)

    try:
        os.remove(raw_file)
    except OSError:
        pass

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        raise FileNotFoundError(
            f"[transcript_engine] Audio conversion failed for {video_id}"
        )

    logger.info(f"[transcript_engine] Audio ready: {wav_path}")
    return wav_path


def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Re-encode *input_path* to a 16 kHz, 1-channel PCM WAV at *output_path*."""
    cmd = [
        _ffmpeg(), "-y",
        "-i",  input_path,
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[transcript_engine] ffmpeg conversion failed.\n"
            f"  Input : {input_path}\n"
            f"  Stderr: {result.stderr[-800:]}"
        )


# ── Audio duration helper ─────────────────────────────────────────────────────

def _get_audio_duration(wav_path: str) -> float:
    """Return the duration of *wav_path* in seconds."""
    try:
        ffprobe = _ffmpeg().replace("ffmpeg", "ffprobe").replace("ffmpeg.exe", "ffprobe.exe")
        result = subprocess.run(
            [ffprobe, "-v", "quiet", "-print_format", "json", "-show_streams", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            info = _json.loads(result.stdout)
            for stream in info.get("streams", []):
                if "duration" in stream:
                    return float(stream["duration"])
    except Exception:
        pass

    try:
        with wave.open(wav_path, "r") as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception as exc:
        logger.warning(f"[transcript_engine] Could not determine audio duration: {exc}")
        return 0.0


def _extract_audio_chunk(wav_path: str, start: float, end: float, idx: int) -> str:
    """Slice *wav_path* from *start* to *end* seconds into a temporary chunk file."""
    chunk_path = wav_path.replace(".wav", f"_chunk{idx}.wav")
    cmd = [
        _ffmpeg(), "-y",
        "-i",  wav_path,
        "-ss", str(start),
        "-to", str(end),
        "-ar", "16000",
        "-ac", "1",
        chunk_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"[transcript_engine] Audio chunk extraction failed.\n"
            f"  Stderr: {result.stderr[-400:]}"
        )
    return chunk_path


# ── Whisper fallback ──────────────────────────────────────────────────────────

def _whisper_segments_to_standard(raw_segments: list, offset: float = 0.0) -> list[dict]:
    """Convert Whisper segment dicts to the standard {start, end, text} format."""
    out = []
    for seg in raw_segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        out.append({
            "start": round(float(seg["start"]) + offset, 3),
            "end":   round(float(seg["end"])   + offset, 3),
            "text":  text,
        })
    return out


def fallback_whisper_transcription(video_url: str) -> list[dict]:
    """
    Transcribe the audio of *video_url* using OpenAI Whisper.

    Automatically detects non-English speech.
    Chunks audio into CHUNK_MINUTES-minute segments for long videos.

    Returns:
        List of {"start": float, "end": float, "text": str}.

    Raises:
        ImportError: if openai-whisper or torch are not installed.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is required for the ASR fallback. "
            "Install with: pip install openai-whisper"
        )

    import torch

    wav_path = extract_audio(video_url)
    device   = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        f"[transcript_engine] Loading Whisper '{WHISPER_MODEL_SIZE}' on {device} …"
    )
    model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)

    duration       = _get_audio_duration(wav_path)
    chunk_seconds  = CHUNK_MINUTES * 60
    all_segments: list[dict] = []

    if duration <= chunk_seconds or duration == 0.0:
        logger.info("[transcript_engine] Transcribing with Whisper (single pass) …")
        result = model.transcribe(wav_path, word_timestamps=True)
        all_segments = _whisper_segments_to_standard(result["segments"])
    else:
        num_chunks = math.ceil(duration / chunk_seconds)
        logger.info(
            f"[transcript_engine] Long video ({duration:.0f}s) — "
            f"chunking into {num_chunks} × {CHUNK_MINUTES}-min segments …"
        )
        for i in range(num_chunks):
            t_start = i * chunk_seconds
            t_end   = min(t_start + chunk_seconds, duration)
            logger.info(
                f"[transcript_engine]   Chunk {i + 1}/{num_chunks}: "
                f"{t_start:.0f}s – {t_end:.0f}s"
            )
            chunk_path = _extract_audio_chunk(wav_path, t_start, t_end, i)
            try:
                result = model.transcribe(chunk_path, word_timestamps=True)
                all_segments.extend(
                    _whisper_segments_to_standard(result["segments"], offset=t_start)
                )
            finally:
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

    logger.info(
        f"[transcript_engine] Whisper transcription complete: "
        f"{len(all_segments)} segments"
    )
    return all_segments


# ── Main public entry point ───────────────────────────────────────────────────

def get_transcript(video_url: str) -> list[dict]:
    """
    Hybrid transcript acquisition for any YouTube URL.

    Strategy:
      1. Try youtube-transcript-api (prefers manual captions, accepts auto-CC).
      2. If unavailable or empty → run Whisper ASR fallback locally.

    Args:
        video_url: Any valid YouTube URL.

    Returns:
        Non-empty list of {"start": float, "end": float, "text": str} dicts.

    Raises:
        ValueError:   if the video ID cannot be parsed.
        RuntimeError: if both strategies fail.
    """
    video_id = _get_video_id(video_url)
    if not video_id:
        raise ValueError(f"[transcript_engine] Cannot parse video ID from: {video_url!r}")

    # ── Strategy 1: YouTube Transcript API ────────────────────────────────────
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        logger.info(
            f"[transcript_engine] Trying YouTube transcript API for {video_id} …"
        )
        api        = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)
        segments   = _segments_from_api(transcript)

        if segments:
            logger.info(
                f"[transcript_engine] YouTube API OK — {len(segments)} segments"
            )
            return segments

        logger.warning(
            "[transcript_engine] YouTube API returned an empty transcript — "
            "falling back to Whisper."
        )

    except Exception as exc:
        logger.warning(
            f"[transcript_engine] YouTube transcript API failed "
            f"({type(exc).__name__}: {exc}) — falling back to Whisper."
        )

    # ── Strategy 2: Whisper ASR ───────────────────────────────────────────────
    logger.info("[transcript_engine] Activating Whisper ASR fallback …")
    segments = fallback_whisper_transcription(video_url)

    if not segments:
        raise RuntimeError(
            f"[transcript_engine] Both transcript strategies failed for {video_id}. "
            "The video may have no audio or an unsupported format."
        )

    return segments
