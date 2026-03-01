"""
src/detection.py
Runs YOLOv8s on extracted frames using Ultralytics.
Model auto-downloads on first use — no local .pt files required.
Returns structured detection results per timestamp.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# COCO class names relevant to event building
RELEVANT_CLASSES = {
    "person", "sports ball", "microphone", "trophy", "laptop",
    "cell phone", "book", "chair", "car", "bicycle", "dog", "cat",
    "bottle", "cup", "keyboard", "mouse", "remote", "tv", "clock",
    "handbag", "backpack", "umbrella", "tie", "suitcase",
}

MIN_BOX_AREA = 0.015      # 1.5% of frame area minimum
PERSON_TRACK_ONLY = False  # track ALL classes for full motion intelligence


@dataclass
class Detection:
    class_name: str
    confidence: float
    box_xyxy: list[float]           # [x1, y1, x2, y2] normalised 0-1
    track_id: Optional[int] = None  # filled in by tracker


@dataclass
class FrameDetections:
    timestamp: float
    detections: list[Detection] = field(default_factory=list)

    @property
    def object_names(self) -> list[str]:
        return [d.class_name for d in self.detections]

    @property
    def person_boxes(self) -> list[list[float]]:
        return [d.box_xyxy for d in self.detections if d.class_name == "person"]


class ObjectDetector:
    """
    Wraps Ultralytics YOLOv8s for frame-level object detection.
    Model is loaded lazily and reused across all frames.
    Ultralytics will auto-download the model weights on first use.
    """

    MODEL_NAME = "yolov8s.pt"  # better accuracy, still CPU safe; auto-downloaded by Ultralytics

    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                logger.info(f"Loading YOLO model: {self.MODEL_NAME} (auto-download if needed)")
                self._model = YOLO(self.MODEL_NAME)
                logger.info("YOLO model loaded.")
            except ImportError:
                raise ImportError(
                    "ultralytics is required: pip install ultralytics"
                )

    def detect(self, frame: np.ndarray, timestamp: float) -> FrameDetections:
        """
        Run YOLOv8 on a single frame.

        Args:
            frame: BGR numpy array.
            timestamp: Timestamp in seconds for this frame.

        Returns:
            FrameDetections with all detected objects.
        """
        self._load_model()

        results = self._model.track(
            frame,
            persist=True,
            imgsz=640,
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        frame_h, frame_w = frame.shape[:2]

        detections: list[Detection] = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            norm_box = [
                x1 / frame_w, y1 / frame_h,
                x2 / frame_w, y2 / frame_h,
            ]

            # Area filter: skip tiny boxes (< 1.5% of frame area)
            box_area = (norm_box[2] - norm_box[0]) * (norm_box[3] - norm_box[1])
            if box_area < MIN_BOX_AREA:
                continue

            # Person-only tracking: suppress tracker noise on other classes
            track_id = None
            if box.id is not None and (not PERSON_TRACK_ONLY or class_name == "person"):
                track_id = int(box.id[0])

            detections.append(Detection(
                class_name=class_name,
                confidence=round(confidence, 3),
                box_xyxy=norm_box,
                track_id=track_id,
            ))

        return FrameDetections(timestamp=timestamp, detections=detections)

    def detect_batch(
        self,
        frames: list[tuple[float, np.ndarray]],
        batch_size: int = 4,
    ) -> list[FrameDetections]:
        """
        Detect objects in a batch of frames efficiently.

        Args:
            frames: List of (timestamp, frame) tuples.
            batch_size: How many frames to process at once.

        Returns:
            List of FrameDetections ordered by timestamp.
        """
        self._load_model()
        logger.info(f"detect_batch: processing {len(frames)} frames (batch_size={batch_size})")
        results_list: list[FrameDetections] = []

        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            timestamps = [t for t, _ in batch]
            images = [f for _, f in batch]

            batch_results = self._model.track(
                images,
                persist=True,
                imgsz=640,
                conf=self.confidence_threshold,
                verbose=False,
            )

            for ts, img, result in zip(timestamps, images, batch_results):
                frame_h, frame_w = img.shape[:2]
                detections = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    norm_box = [
                        x1 / frame_w, y1 / frame_h,
                        x2 / frame_w, y2 / frame_h,
                    ]

                    box_area = (norm_box[2] - norm_box[0]) * (norm_box[3] - norm_box[1])
                    if box_area < MIN_BOX_AREA:
                        continue

                    track_id = None
                    if box.id is not None and (not PERSON_TRACK_ONLY or class_name == "person"):
                        track_id = int(box.id[0])

                    detections.append(Detection(
                        class_name=class_name,
                        confidence=round(confidence, 3),
                        box_xyxy=norm_box,
                        track_id=track_id,
                    ))
                results_list.append(FrameDetections(timestamp=ts, detections=detections))

        return results_list
