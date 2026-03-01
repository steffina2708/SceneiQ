"""
src/downloader.py
Downloads YouTube videos using the yt_dlp Python API.
No subprocess, no PATH dependency, no ffmpeg required.
"""

from pathlib import Path
from yt_dlp import YoutubeDL


class VideoDownloader:
    def __init__(self, output_dir="uploads"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str) -> dict:
        """
        Downloads a YouTube video safely using yt_dlp Python API.
        Prefers progressive MP4 (video+audio combined) to avoid ffmpeg merging.
        Returns metadata including local file path.
        """
        try:
            ydl_opts = {
                # Prefer progressive MP4 (audio+video combined)
                "format": "best[ext=mp4][protocol=https]",
                "outtmpl": str(self.output_dir / "%(id)s.%(ext)s"),
                "quiet": True,
                "noplaylist": True,
                "merge_output_format": "mp4",
            }

            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                video_id = info.get("id")
                ext = info.get("ext", "mp4")
                filename = self.output_dir / f"{video_id}.{ext}"

                # Fallback in case extension differs
                if not filename.exists():
                    for file in self.output_dir.glob(f"{video_id}.*"):
                        filename = file
                        break

                if not filename.exists():
                    raise FileNotFoundError("Downloaded file not found after yt-dlp execution.")

                return {
                    "status": "success",
                    "video_id": video_id,
                    "title": info.get("title"),
                    "file_path": str(filename),
                }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }
