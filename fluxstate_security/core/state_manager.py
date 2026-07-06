# core/state_manager.py
import time
from enum import Enum

class IntelligenceState(Enum):
    IDLE = "MONITORING_ENVIRONMENT"
    OBSERVING = "TRACKING_KINETIC_ENTITIES"
    REASONING = "GENERATING_SEMANTIC_CONTEXT"

class RealityGraph:
    """
    Instead of hardcoded security rules (Access Granted/Breach),
    this graph builds a semantic history of the physical space.
    It logs contextual events rather than making arbitrary assumptions.
    """
    def __init__(self):
        self.current_state = IntelligenceState.IDLE
        self.event_log = []
        self.last_event_time = time.time()

    def set_state(self, new_state: IntelligenceState):
        self.current_state = new_state

    def log_event(self, context_summary: str):
        """Logs a natural language observation of the physical space."""
        # Avoid spamming the exact same event consecutively
        if len(self.event_log) > 0 and self.event_log[-1]["summary"] == context_summary:
            return 
            
        print(f"\n[REALITY GRAPH] Observation: {context_summary}")
        self.event_log.append({
            "timestamp": time.time(),
            "summary": context_summary
        })
        self.last_event_time = time.time()
        
    def get_recent_events(self, limit=5):
        """Returns the most recent events for UI rendering."""
        return self.event_log[-limit:]

import collections
import threading
import uuid
import os
import cv2

class EpisodicMemoryBuffer:
    """
    V2 Adaptive Architecture: Episodic Memory.
    Stores the last 10 minutes (configurable) of scene state for delayed operator feedback.
    Supports O(1) append operations, thread-safe access, and timestamp-based auto-eviction.
    Minimizes RAM footprint by persisting frames to disk and storing only references.
    """
    def __init__(self, ttl_seconds=600, max_size=1000, temp_frame_dir="/tmp/flux_episodic"):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.buffer = collections.deque()
        self.lock = threading.Lock()
        self.temp_frame_dir = temp_frame_dir
        
        # Ensure tmp directory exists
        os.makedirs(self.temp_frame_dir, exist_ok=True)

    def _evict_old(self, current_time: float):
        """Must be called under lock. Removes expired or overflowing episodes."""
        # Evict by TTL
        while self.buffer and (current_time - self.buffer[0]['timestamp']) > self.ttl_seconds:
            self._remove_leftmost()
            
        # Evict by max capacity (prevent unbounded growth if burst of events)
        while len(self.buffer) > self.max_size:
            self._remove_leftmost()

    def _remove_leftmost(self):
        """Removes oldest item and cleans up its file reference."""
        item = self.buffer.popleft()
        if item.get("frame_path") and os.path.exists(item["frame_path"]):
            try:
                os.remove(item["frame_path"])
            except OSError:
                pass

    def add_episode(self, rule_triggered: str, scene_description: str, telemetry: dict, frame=None) -> str:
        """
        O(1) append operation. 
        Saves frame to disk (if provided) to prevent memory ballooning, stores metadata.
        """
        current_time = time.time()
        event_id = uuid.uuid4().hex
        
        frame_path = None
        if frame is not None:
            frame_path = os.path.join(self.temp_frame_dir, f"{event_id}.jpg")
            try:
                # Save at 70% quality to save space/time
                cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            except Exception:
                frame_path = None
                
        episode = {
            "timestamp": current_time,
            "event_id": event_id,
            "rule_triggered": rule_triggered,
            "scene_description": scene_description,
            "telemetry": telemetry,
            "frame_path": frame_path
        }
        
        with self.lock:
            self.buffer.append(episode)
            self._evict_old(current_time)
            
        return event_id

    def get_episode(self, event_id: str) -> dict:
        """O(N) lookup for specific feedback mapping. Since N is small, this is extremely fast."""
        with self.lock:
            # We don't necessarily evict on read to keep lookup fast, but we can do a lazy check
            for ep in self.buffer:
                if ep["event_id"] == event_id:
                    return dict(ep)
        return None

    def get_recent_episodes(self, limit=10) -> list:
        """Returns shallow copies of recent episodes."""
        with self.lock:
            self._evict_old(time.time())
            return [dict(ep) for ep in list(self.buffer)[-limit:]]
            
    def get_size(self) -> int:
        with self.lock:
            return len(self.buffer)
            
    def clear(self):
        """Purges the entire buffer and deletes associated files."""
        with self.lock:
            while self.buffer:
                self._remove_leftmost()
