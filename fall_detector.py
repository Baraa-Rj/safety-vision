"""Live fall detection: best.pt tracking -> per-track window -> FallTCN.

Confirmation gate: a fall is emitted only after `confirm` consecutive positive
windows, which suppresses the brief drop-like signal from bending/crouching and
controls the false-positive rate. A per-track cooldown prevents duplicate
alerts for the same event.

Integration: pass an object exposing processDetection(result) (your
AlertManager). If omitted, a callback is invoked instead.
"""
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import torch
from ultralytics import YOLO

from features import TrackBuffer
from model import load_fall_model, standardize


@dataclass
class DetectionResult:
    type: str = "FALL_DETECTED"
    workerId: int = -1            # tracker id; map to QR worker id upstream (-1 = UNIDENTIFIED)
    zoneId: Optional[int] = None
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    bbox: Optional[tuple] = None  # (x1, y1, x2, y2)
    frame: Optional[object] = None  # snapshot for SR2.2; numpy image or None


class FallDetector:
    def __init__(self, yolo_weights, fall_model, alert_manager=None,
                 on_fall: Optional[Callable[[DetectionResult], None]] = None,
                 conf=0.3, confirm=3, cooldown_s=10.0, max_gap=5,
                 tracker="bytetrack.yaml", device=None, prune_after=300):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(yolo_weights)
        self.person_cls = self._resolve_person_cls()
        self.model, ck = load_fall_model(fall_model, self.device)
        self.window = ck["window"]
        self.mean, self.std, self.threshold = ck["mean"], ck["std"], ck["threshold"]
        self.alert_manager = alert_manager
        self.on_fall = on_fall
        self.conf = conf
        self.confirm = confirm
        self.cooldown_s = cooldown_s
        self.max_gap = max_gap
        self.tracker = tracker
        self.prune_after = prune_after        # frames a track may be unseen before eviction
        self._buffers = {}
        self._pos_streak = {}
        self._cooldown_until = {}
        self._last_seen = {}

    def _resolve_person_cls(self):
        for k, v in self.yolo.names.items():
            if str(v).lower() == "person":
                return int(k)
        raise ValueError(f"No 'person' class in {self.yolo.names}")

    def _infer(self, window_arr):
        x = standardize(window_arr, self.mean, self.std)[None]  # [1, T, F]
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x).float().to(self.device))
            return float(torch.softmax(logits, 1)[0, 1])

    def _emit(self, tid, prob, box, frame):
        result = DetectionResult(workerId=int(tid), confidence=prob,
                                 bbox=tuple(map(float, box)), frame=frame)
        if self.alert_manager is not None:
            self.alert_manager.processDetection(result)
        if self.on_fall is not None:
            self.on_fall(result)
        return result

    def _prune(self, frame_idx):
        """Evict per-track state for ids not seen for `prune_after` frames so the
        dicts stay bounded over long live runs (ByteTrack ids grow forever)."""
        stale = [t for t, f in self._last_seen.items() if frame_idx - f > self.prune_after]
        for t in stale:
            self._buffers.pop(t, None)
            self._pos_streak.pop(t, None)
            self._cooldown_until.pop(t, None)
            self._last_seen.pop(t, None)

    def update(self, boxes, ids, frame_idx, frame_h, frame=None, now=None):
        """Process one frame's person boxes + track ids; return any falls.

        Use this to integrate with the existing pipeline's tracker instead of
        run() — pass the per-frame person boxes/ids the PPE path already produces,
        so best.pt is not tracked a second time.

        boxes: iterable of (x1, y1, x2, y2); ids: matching track ids.
        Returns a (usually empty) list of DetectionResult.
        """
        now = now if now is not None else time.time()
        out = []
        for box, tid in zip(boxes, ids):
            tid = int(tid)
            self._last_seen[tid] = frame_idx
            buf = self._buffers.setdefault(tid, TrackBuffer(self.window, self.max_gap))
            buf.update(tuple(box), frame_idx, frame_h)
            if not buf.ready():
                continue
            prob = self._infer(buf.array())
            streak = self._pos_streak.get(tid, 0)
            streak = streak + 1 if prob >= self.threshold else 0
            self._pos_streak[tid] = streak
            if streak >= self.confirm and now >= self._cooldown_until.get(tid, 0):
                self._cooldown_until[tid] = now + self.cooldown_s
                self._pos_streak[tid] = 0
                out.append(self._emit(tid, prob, box, frame))
        self._prune(frame_idx)
        return out

    def run(self, source):
        """Process a video file or RTSP stream; yields DetectionResult per fall.

        Standalone path (owns its own tracker). For live-pipeline integration
        call update() per frame instead, to avoid tracking best.pt twice.
        """
        results = self.yolo.track(
            source=source, stream=True, persist=True, conf=self.conf,
            tracker=self.tracker, classes=[self.person_cls], verbose=False,
        )
        for frame_idx, r in enumerate(results):
            frame_h = r.orig_shape[0]
            if r.boxes is None or r.boxes.id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.cpu().numpy().astype(int)
            for ev in self.update(xyxy, ids, frame_idx, frame_h,
                                  frame=getattr(r, "orig_img", None)):
                yield ev


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--fall-model", default="fall_model.pt")
    ap.add_argument("--source", required=True)  # video path or rtsp://...
    args = ap.parse_args()

    det = FallDetector(args.weights, args.fall_model)
    for ev in det.run(args.source):
        print(f"FALL  track={ev.workerId}  conf={ev.confidence:.2f}  bbox={ev.bbox}")
