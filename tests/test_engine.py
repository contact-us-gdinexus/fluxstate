import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set environment variable to enable testing adaptive mode in some cases
os.environ["ADAPTIVE_INTELLIGENCE"] = "False"
from fluxstate_security.core.engine import FluxInferenceEngine

class MockRetriever:
    def get_contextual_memories(self, scene_description, top_k):
        return []

class MockBuilder:
    def build_prompt(self, current_scene_description, rule_triggered, retrieved_memories):
        return "Prompt"

class MockPool:
    def investigate_scene(self, context_log=None, frame=None, prompt=None, **kwargs):
        return "ADAPTIVE: FALSE POSITIVE"

class TestEngine(unittest.TestCase):
    def setUp(self):
        os.environ["ADAPTIVE_INTELLIGENCE"] = "False"
        self.engine = FluxInferenceEngine()
        
    def test_legacy_mode(self):
        # Trigger KINETIC_AGGRESSION by setting action="FIGHTING" and acoustic="SCREAM"
        # We need to mock _load_policy to guarantee KINETIC_AGGRESSION is enabled
        self.engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
        
        detected_objects = [(0, 0, 10, 10, "person", 0.99, 1)]
        res, _, _ = self.engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
        self.assertIn("THREAT VECTOR", res)
        self.assertNotIn("ADAPTIVE", res)

    def test_adaptive_mode_suppression(self):
        os.environ["ADAPTIVE_INTELLIGENCE"] = "True"
        engine = FluxInferenceEngine()
        # Mock adaptive components
        engine.adaptive_mode = True
        engine.retriever = MockRetriever()
        engine.builder = MockBuilder()
        engine.pool = MockPool()
        
        engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
        detected_objects = [(0, 0, 10, 10, "person", 0.99, 1)]
        
        # Test suppression
        res, _, _ = engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
        self.assertIn("ADAPTIVE: SUPPRESSED", res)
        self.assertNotIn("THREAT VECTOR", res) # Successfully suppressed!

    def test_adaptive_mode_confirmation(self):
        os.environ["ADAPTIVE_INTELLIGENCE"] = "True"
        engine = FluxInferenceEngine()
        engine.adaptive_mode = True
        engine.retriever = MockRetriever()
        engine.builder = MockBuilder()
        engine.pool = MockPool()
        engine.pool.investigate_scene = lambda **kwargs: "I SEE A WEAPON, DEFINITE THREAT"
        
        engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
        detected_objects = [(0, 0, 10, 10, "person", 0.99, 1)]
        
        res, _, _ = engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
        self.assertIn("THREAT VECTOR", res)
        self.assertIn("ADAPTIVE: CONFIRMED", res)

    def test_adaptive_fallback_retriever_failure(self):
        os.environ["ADAPTIVE_INTELLIGENCE"] = "True"
        engine = FluxInferenceEngine()
        engine.adaptive_mode = True
        
        class FailingRetriever:
            def get_contextual_memories(self, a, top_k):
                raise Exception("Retriever died")
                
        engine.retriever = FailingRetriever()
        engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
        detected_objects = [(0, 0, 10, 10, "person", 0.99, 1)]
        
        res, _, _ = engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
        # Should gracefully fall back to legacy output
        self.assertIn("THREAT VECTOR", res)
        self.assertNotIn("ADAPTIVE", res)

    def test_adaptive_fallback_vlm_failure(self):
        os.environ["ADAPTIVE_INTELLIGENCE"] = "True"
        engine = FluxInferenceEngine()
        engine.adaptive_mode = True
        engine.retriever = MockRetriever()
        engine.builder = MockBuilder()
        
        class FailingPool:
            def investigate_scene(self, **kwargs):
                raise Exception("VLM OOM")
                
        engine.pool = FailingPool()
        engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
        detected_objects = [(0, 0, 10, 10, "person", 0.99, 1)]
        
        res, _, _ = engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
        self.assertIn("THREAT VECTOR", res)
        self.assertNotIn("ADAPTIVE", res)

def run_benchmark():
    print("\\n--- Running Engine Integration Benchmarks ---")
    
    # 1. Legacy Mode Baseline
    os.environ["ADAPTIVE_INTELLIGENCE"] = "False"
    legacy_engine = FluxInferenceEngine()
    legacy_engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
    
    detected_objects = [(0, 0, 10, 10, "person", 0.99, 1)]
    
    # Warmup
    legacy_engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
    
    t0 = time.time()
    for _ in range(100):
        legacy_engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
    legacy_time = (time.time() - t0) / 100 * 1000 - 50 # Subtract the 50ms sleep
    print(f"Legacy Engine Latency Overhead: {legacy_time:.2f} ms")
    if legacy_time < 2.0:
         print("[PASS] Legacy overhead < 2% / 2ms")
         
    # 2. Adaptive Mode End-to-End
    os.environ["ADAPTIVE_INTELLIGENCE"] = "True"
    adaptive_engine = FluxInferenceEngine()
    adaptive_engine._load_policy = lambda: {"THREAT_VECTORS": {"KINETIC_AGGRESSION": {"enabled": True, "trigger_actions": ["FIGHTING"], "trigger_acoustics": ["SCREAM"]}}}
    
    # Adaptive Mock for deterministic timing (skip real VLM weights since it's an integration test framework)
    adaptive_engine.adaptive_mode = True
    adaptive_engine.retriever = MockRetriever()
    adaptive_engine.builder = MockBuilder()
    adaptive_engine.pool = MockPool()
    
    t0 = time.time()
    for _ in range(10):
        adaptive_engine.generate_context_reasoning(detected_objects, [], "FIGHTING", acoustic_event="SCREAM")
    adaptive_time = (time.time() - t0) / 10 * 1000 - 50
    print(f"Adaptive Engine Orchestration Latency: {adaptive_time:.2f} ms (Excluding raw hardware inference time)")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        run_benchmark()
    else:
        unittest.main()
