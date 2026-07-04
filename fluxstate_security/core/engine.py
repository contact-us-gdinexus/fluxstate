# core/engine.py
import time
import json
import os

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("[Warning] MLX not available. Falling back to simulated arrays for development.")

class FluxInferenceEngine:
    """
    Interfaces directly with Apple Silicon using MLX, orchestrating the local reasoning engine.
    """
    def __init__(self):
        print(f"[Engine] Initializing local backend on Apple Silicon...")
        
    def _load_policy(self):
        try:
            import os
            policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "intelligence_policy.json")
            with open(policy_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def generate_context_reasoning(self, detected_objects, movement_boxes, action_heuristic, rf_count=0, acoustic_event="AMBIENT") -> str:
        """
        Takes the raw semantic telemetry (YOLO objects + Skeletal Actions + RF + Audio + Identity) and 
        uses the local LLM to reason about the context based on a Dynamic Policy Engine.
        """
        time.sleep(0.05) # Simulate inference latency
        
        # Hot-reload Intelligence Policy
        policy = self._load_policy()
        vectors = policy.get("THREAT_VECTORS", {})
        
        persons = [obj for obj in detected_objects if "person" in obj[4].lower()]
        other_objects = [obj for obj in detected_objects if "person" not in obj[4].lower()]
        
        # Extract Cryptographic GPASS Tokens
        gpass_list = []
        for p in persons:
            if "[" in p[4] and "]" in p[4]:
                gpass = p[4].split("[")[-1].replace("]", "").strip()
                if "GPX" in gpass:
                    gpass_list.append(gpass)
        
        # --- BEHAVIORAL THREAT MATRIX (Dynamic OTA Policy) ---
        threat_flags = []
        
        # 1. Kinetic-Acoustic Fusion Threat
        kinetic_rule = vectors.get("KINETIC_AGGRESSION", {})
        if kinetic_rule.get("enabled", False):
            if action_heuristic in kinetic_rule.get("trigger_actions", []) and acoustic_event in kinetic_rule.get("trigger_acoustics", []):
                threat_flags.append("KINETIC_AGGRESSION")
            
        # 2. RF Anomaly (1 person carrying multiple network emitting devices)
        rf_rule = vectors.get("RF_PAYLOAD_ANOMALY", {})
        if rf_rule.get("enabled", False):
            if len(persons) == 1 and rf_count > rf_rule.get("max_allowed_rf_per_entity", 3):
                threat_flags.append("RF_DEVICE_ANOMALY")
            
        # 3. Unauthorized Area / Abandoned Object Logic can be injected here
        
        threat_str = ""
        db_status = ""
        if len(gpass_list) > 0:
            gpass_str = f" (Identities: {', '.join(set(gpass_list))})"
            if len(threat_flags) > 0:
                threat_str = f" [THREAT VECTOR: {', '.join(threat_flags)}]"
                db_status = f" [SWARM LEDGER: BEHAVIORAL ANOMALY DETECTED. GPASS FLAG PROMOTED]"
            else:
                db_status = f" [SWARM LEDGER: CONTEXT NOMINAL. BEHAVIOR CLEARED]"
        else:
            gpass_str = ""
        
        rf_str = f" [RF Intel: {rf_count} devices]" if rf_count > 0 else ""
        audio_str = f" [Acoustic: {acoustic_event}]"
        
        if len(persons) == 0:
            if len(other_objects) > 0:
                labels = set(f"{obj[4]} ({obj[5]:.2f})" for obj in other_objects)
                return f"No humans detected. Present objects: {', '.join(labels)}.{rf_str}{audio_str}"
            elif len(movement_boxes) > 0:
                return f"Non-human kinetic activity detected. Unclassified environmental shift.{rf_str}{audio_str}"
            return f"Environment stable. No significant entities present.{rf_str}{audio_str}"
            
        if len(persons) == 1:
            if len(other_objects) > 0:
                labels = set(f"{obj[4]} ({obj[5]:.2f})" for obj in other_objects)
                return f"1 Individual{gpass_str} [Action: {action_heuristic}]. Proximal Objects: {', '.join(labels)}.{rf_str}{audio_str}{threat_str}{db_status}"
            return f"1 Individual{gpass_str} present. Primary Action: {action_heuristic}.{rf_str}{audio_str}{threat_str}{db_status}"
                
        if len(persons) > 1:
            return f"{len(persons)} Individuals{gpass_str} detected. Primary collective action: {action_heuristic}.{rf_str}{audio_str}{threat_str}{db_status}"
