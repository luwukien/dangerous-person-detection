"""Per-video tracking and temporal labels for the two-class person detector.

Vote shares measure agreement among detections, NOT class probabilities.
Create a new VideoTracker for each video; no model-global tracking state is used.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from types import SimpleNamespace

import numpy as np


@dataclass
class LabelState:
    label: int | None = None
    last_seen: int = 0
    votes: deque = field(default_factory=deque)


class TemporalLabels:
    def __init__(self, fps, window_seconds=0.6, min_seconds=0.2,
                 forget_seconds=0.5, agreement=0.75):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        if not 0 < min_seconds <= window_seconds or forget_seconds <= 0:
            raise ValueError("Invalid temporal durations")
        if not 0.5 < agreement <= 1:
            raise ValueError("agreement must be in (0.5, 1]")
        self.window = max(1, math.ceil(fps * window_seconds))
        self.minimum = max(1, math.ceil(fps * min_seconds))
        self.max_age = max(1, math.ceil(fps * forget_seconds))
        self.agreement = agreement
        self.states = {}

    def expire(self, frame_index):
        self.states = {key: state for key, state in self.states.items()
                       if frame_index - state.last_seen <= self.max_age}

    def update(self, track_id, class_id, confidence, frame_index):
        if class_id not in (0, 1) or not 0 < confidence <= 1:
            raise ValueError("Expected binary class and confidence in (0, 1]")
        self.expire(frame_index)
        state = self.states.setdefault(track_id, LabelState())
        # A short gap retains the displayed label but resets supporting votes.
        if state.last_seen != frame_index - 1:
            state.votes.clear()
        state.last_seen = frame_index
        state.votes.append((class_id, confidence))
        while len(state.votes) > self.window:
            state.votes.popleft()
        if len(state.votes) >= self.minimum:
            total = sum(conf for _, conf in state.votes)
            positive = sum(conf for cls, conf in state.votes if cls == 1) / total
            if positive >= self.agreement:
                state.label = 1
            elif positive <= 1 - self.agreement:
                state.label = 0
        return state.label


class VideoTracker:
    """Class-independent ByteTrack association followed by per-ID voting."""

    def __init__(self, fps):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.labels = TemporalLabels(fps)
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=0.5, track_low_thresh=0.1,
            new_track_thresh=0.5, track_buffer=self.labels.max_age,
            match_thresh=0.8, fuse_score=True,
        ))
        self.frame_index = 0
        self.observations = []
        self.previous = {}
        self.raw_switches = 0
        self.stable_switches = 0

    def annotate(self, result):
        self.frame_index += 1
        self.labels.expire(self.frame_index)
        self.previous = {key: value for key, value in self.previous.items()
                         if self.frame_index - value[2] <= self.labels.max_age}
        tracks = self.tracker.update(result.boxes.cpu().numpy(), result.orig_img)
        rows = []
        self.observations = []
        for track in tracks:
            x1, y1, x2, y2, track_id, confidence, raw_class, _ = track
            track_id, raw_class = int(track_id), int(raw_class)
            stable = self.labels.update(track_id, raw_class, float(confidence), self.frame_index)
            old = self.previous.get(track_id)
            if old is not None:
                self.raw_switches += int(old[0] != raw_class)
                self.stable_switches += int(old[1] is not None and stable is not None and old[1] != stable)
            self.previous[track_id] = (raw_class, stable, self.frame_index)
            rows.append([x1, y1, x2, y2, track_id, confidence, 2 if stable is None else stable])
            self.observations.append((self.frame_index, track_id, raw_class, float(confidence), stable))
        annotated = result.new()
        annotated.names = dict(result.names)
        annotated.names[2] = "pending"
        annotated.update(boxes=np.asarray(rows, dtype=np.float32).reshape(-1, 7))
        # Raw confidence belongs to the raw class, not a retained temporal label.
        return annotated.plot(conf=False)
