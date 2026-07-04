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
