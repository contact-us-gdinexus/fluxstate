import time
import json
import os
import logging
import numpy as np

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FluxEngine")

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("[Warning] MLX not available. Falling back to simulated arrays for development.")

class FluxInferenceEngine:
    """
    Interfaces directly with Apple Silicon using MLX, orchestrating the local reasoning engine.
    Integrates the Adaptive Intelligence Semantic RAG pipeline.
    """
    def __init__(self, db=None):
        logger.info("[Engine] Initializing local backend on Apple Silicon...")
        
        # Feature flag controls the entire adaptive pipeline
        self.adaptive_mode = os.environ.get("ADAPTIVE_INTELLIGENCE", "False").lower() in ("true", "1")
        
        self.pool = None
        self.retriever = None
        self.builder = None
        self.db = db
        
        if self.adaptive_mode:
            logger.info("[Engine] ADAPTIVE_INTELLIGENCE is True. Loading RAG infrastructure.")
            try:
                # Lazy import adaptive components to prevent overhead in legacy mode
                from .agent import MLXModelPool
                from .forensics import ForensicDatabase
                from .adaptation.retriever import SemanticRetriever
                from .adaptation.prompt_builder import PromptBuilder
                
                if self.db is None:
                    self.db = ForensicDatabase()
                    
                self.pool = MLXModelPool(enable_adaptive=True)
                self.retriever = SemanticRetriever(db=self.db, agent_pool=self.pool, similarity_threshold=0.85)
                self.builder = PromptBuilder(max_memories=3, min_similarity=0.85)
            except Exception as e:
                logger.error(f"[Engine] Failed to load adaptive components: {e}. Automatically falling back to Legacy mode.")
                self.adaptive_mode = False

    def _load_policy(self):
        try:
            policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "intelligence_policy.json")
            with open(policy_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def generate_context_reasoning(self, detected_objects, movement_boxes, action_heuristic, rf_count=0, acoustic_event="AMBIENT", frame=None):
        """
        Takes the raw semantic telemetry and uses the local LLM to reason about the context based on a Dynamic Policy Engine.
        Returns: (context_string, vlm_reasoning, adaptive_status)
        """
        start_total = time.time()
        
        # Simulate slight physical sensor inference latency
        time.sleep(0.05) 
        
        # --- 1. LEGACY RULE ENGINE ---
        policy = self._load_policy()
        vectors = policy.get("THREAT_VECTORS", {})
        
        persons = [obj for obj in detected_objects if "person" in obj[4].lower()]
        other_objects = [obj for obj in detected_objects if "person" not in obj[4].lower()]
        
        gpass_list = []
        for p in persons:
            if "[" in p[4] and "]" in p[4]:
                gpass = p[4].split("[")[-1].replace("]", "").strip()
                if "GPX" in gpass:
                    gpass_list.append(gpass)
        
        threat_flags = []
        kinetic_rule = vectors.get("KINETIC_AGGRESSION", {})
        if kinetic_rule.get("enabled", False):
            if action_heuristic in kinetic_rule.get("trigger_actions", []) and acoustic_event in kinetic_rule.get("trigger_acoustics", []):
                threat_flags.append("KINETIC_AGGRESSION")
            
        rf_rule = vectors.get("RF_PAYLOAD_ANOMALY", {})
        if rf_rule.get("enabled", False):
            if len(persons) == 1 and rf_count > rf_rule.get("max_allowed_rf_per_entity", 3):
                threat_flags.append("RF_DEVICE_ANOMALY")
            
        threat_str = ""
        db_status = ""
        is_rule_threat = False
        
        if len(gpass_list) > 0:
            gpass_str = f" (Identities: {', '.join(set(gpass_list))})"
            if len(threat_flags) > 0:
                threat_str = f" [THREAT VECTOR: {', '.join(threat_flags)}]"
                db_status = f" [SWARM LEDGER: BEHAVIORAL ANOMALY DETECTED. GPASS FLAG PROMOTED]"
                is_rule_threat = True
            else:
                db_status = f" [SWARM LEDGER: CONTEXT NOMINAL. BEHAVIOR CLEARED]"
        else:
            gpass_str = ""
            if len(threat_flags) > 0:
                threat_str = f" [THREAT VECTOR: {', '.join(threat_flags)}]"
                is_rule_threat = True
        
        rf_str = f" [RF Intel: {rf_count} devices]" if rf_count > 0 else ""
        audio_str = f" [Acoustic: {acoustic_event}]"
        
        # --- 2. BASE SCENE DESCRIPTION GENERATION ---
        base_scene_desc = ""
        if len(persons) == 0:
            if len(other_objects) > 0:
                labels = set(f"{obj[4]} ({obj[5]:.2f})" for obj in other_objects)
                base_scene_desc = f"No humans detected. Present objects: {', '.join(labels)}.{rf_str}{audio_str}"
            elif len(movement_boxes) > 0:
                base_scene_desc = f"Non-human kinetic activity detected. Unclassified environmental shift.{rf_str}{audio_str}"
            else:
                base_scene_desc = f"Environment stable. No significant entities present.{rf_str}{audio_str}"
        elif len(persons) == 1:
            if len(other_objects) > 0:
                labels = set(f"{obj[4]} ({obj[5]:.2f})" for obj in other_objects)
                base_scene_desc = f"1 Individual{gpass_str} [Action: {action_heuristic}]. Proximal Objects: {', '.join(labels)}.{rf_str}{audio_str}"
            else:
                base_scene_desc = f"1 Individual{gpass_str} present. Primary Action: {action_heuristic}.{rf_str}{audio_str}"
        else:
            base_scene_desc = f"{len(persons)} Individuals{gpass_str} detected. Primary collective action: {action_heuristic}.{rf_str}{audio_str}"

        legacy_output = base_scene_desc + threat_str + db_status
        vlm_reasoning = None
        adaptive_status = "OFF"
        
        # --- 3. ADAPTIVE INTELLIGENCE PIPELINE ---
        if self.adaptive_mode and is_rule_threat:
            try:
                logger.info("[Engine] Adaptive Mode: Processing Rule-Engine Threat through Semantic RAG.")
                
                t_ret = time.time()
                memories = self.retriever.get_contextual_memories(base_scene_desc, top_k=3)
                logger.info(f"[Engine] Retrieval Latency: {(time.time() - t_ret) * 1000:.2f} ms")
                
                t_prompt = time.time()
                prompt = self.builder.build_prompt(
                    current_scene_description=base_scene_desc,
                    rule_triggered=", ".join(threat_flags),
                    retrieved_memories=memories
                )
                logger.info(f"[Engine] Prompt Build Latency: {(time.time() - t_prompt) * 1000:.2f} ms")
                
                t_vlm = time.time()
                # USE THE REAL FRAME. If missing, log warning.
                vlm_frame = frame if frame is not None else np.zeros((10, 10, 3), dtype=np.uint8)
                if frame is None:
                    logger.warning("[Engine] Frame is None. VLM is analyzing an empty frame!")
                    
                vlm_decision = self.pool.investigate_scene(
                    context_log=base_scene_desc,
                    frame=vlm_frame,
                    prompt=prompt
                )
                vlm_reasoning = vlm_decision
                logger.info(f"[Engine] VLM Inference Latency: {(time.time() - t_vlm) * 1000:.2f} ms")
                
                if "FALSE POSITIVE" in vlm_decision.upper() or "BENIGN" in vlm_decision.upper():
                    logger.info("[Engine] Adaptive Intelligence suppressed the alert (False Positive).")
                    adaptive_status = "SUPPRESSED"
                    legacy_output = base_scene_desc + " [ADAPTIVE: SUPPRESSED]"
                else:
                    logger.info("[Engine] Adaptive Intelligence confirmed the threat.")
                    adaptive_status = "CONFIRMED"
                    legacy_output = base_scene_desc + threat_str + " [ADAPTIVE: CONFIRMED] " + vlm_decision
                    
            except Exception as e:
                logger.error(f"[Engine] Adaptive pipeline exception: {e}. Falling back to Legacy mode.")
                adaptive_status = "ERROR"
        
        total_latency = (time.time() - start_total) * 1000
        if self.adaptive_mode and is_rule_threat:
            logger.info(f"[Engine] Total Adaptive Pipeline Latency: {total_latency:.2f} ms")
            
        return legacy_output, vlm_reasoning, adaptive_status
