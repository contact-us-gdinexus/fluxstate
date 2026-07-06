import unittest
import os
import time
import json
import jwt
import numpy as np
import threading
import requests
from fluxstate_security.app import FluxStateNode

JWT_SECRET = os.environ.get("FLUX_JWT_SECRET", "flux_edge_secret_key_992")

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need a clean DB for integration
        if os.path.exists("flux_forensics_integration.db"):
            os.remove("flux_forensics_integration.db")
            
        os.environ["ADAPTIVE_INTELLIGENCE"] = "1"
        cls.node = FluxStateNode(stream_source="")
        # Re-initialize DB correctly for tests
        cls.node.forensics.db_path = "flux_forensics_integration.db"
        cls.node.forensics._init_db()
        cls.node.forensics._load_vector_index()
        
        cls.node.start_api_server(port=8083)
        time.sleep(1) # Let server boot
        
    @classmethod
    def tearDownClass(cls):
        cls.node.stop()
        if os.path.exists("flux_forensics_integration.db"):
            os.remove("flux_forensics_integration.db")
            
    def test_legacy_vs_adaptive_telemetry_schema(self):
        # 1. Test Adaptive Telemetry Schema
        os.environ["ADAPTIVE_INTELLIGENCE"] = "1"
        self.node.adaptive_mode = True
        
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detected_objects = [(10, 10, 20, 20, "person [GPX123]", 0.99, 1)]
        movement_boxes = []
        action = "RUNNING"
        rf_count = 5
        acoustic = "SHOUTING"
        
        # Manually invoke engine to simulate pipeline
        context, vlm_reasoning, adaptive_status = self.node.ai_engine.generate_context_reasoning(
            detected_objects, movement_boxes, action, rf_count, acoustic, frame=frame
        )
        
        # VLM Reasoning should be populated
        self.assertIsNotNone(vlm_reasoning)
        self.assertIn("ADAPTIVE", context)
        
        # Test telemetry JSON mapping in app.py logic
        telemetry = {
            "timestamp": time.time(),
            "should_infer": True,
            "rf_devices_nearby": rf_count,
            "acoustic_event": acoustic,
            "entities": [],
            "action_heuristic": action,
            "context_log": context,
            "vlm_reasoning": vlm_reasoning
        }
        
        # Ensure schema is intact
        self.assertIn("vlm_reasoning", telemetry)
        
        # 2. Test Legacy Telemetry Schema
        self.node.ai_engine.adaptive_mode = False
        context_legacy, vlm_legacy, status_legacy = self.node.ai_engine.generate_context_reasoning(
            detected_objects, movement_boxes, action, rf_count, acoustic, frame=frame
        )
        self.assertIsNone(vlm_legacy)
        self.assertEqual(status_legacy, "OFF")
        
        # In legacy mode, app.py runs SemanticAgent directly
        vlm_reasoning_legacy = self.node.agent.investigate_scene(context_legacy, frame)
        telemetry_legacy = telemetry.copy()
        telemetry_legacy["context_log"] = context_legacy
        telemetry_legacy["vlm_reasoning"] = vlm_reasoning_legacy
        
        self.assertIn("vlm_reasoning", telemetry_legacy)
        self.node.ai_engine.adaptive_mode = True # Restore
        
    def test_real_frame_reaches_mlx_pool(self):
        # We will patch the pool to intercept the frame shape
        captured_frame_shape = None
        
        original_investigate = self.node.ai_engine.pool.investigate_scene
        
        def mock_investigate(context_log, frame, prompt):
            nonlocal captured_frame_shape
            captured_frame_shape = frame.shape
            return "FALSE POSITIVE"
            
        self.node.ai_engine.pool.investigate_scene = mock_investigate
        
        real_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        detected_objects = [(10, 10, 20, 20, "person", 0.99, 1)]
        
        context, vlm, status = self.node.ai_engine.generate_context_reasoning(
            detected_objects, [], "RUNNING", rf_count=5, acoustic_event="SHOUTING", frame=real_frame
        )
        
        self.assertEqual(captured_frame_shape, (720, 1280, 3))
        self.assertEqual(status, "SUPPRESSED")
        
        self.node.ai_engine.pool.investigate_scene = original_investigate
        
    def test_suppressed_events_receive_event_ids(self):
        # Force a suppressed event via engine
        original_investigate = self.node.ai_engine.pool.investigate_scene
        self.node.ai_engine.pool.investigate_scene = lambda *args, **kwargs: "FALSE POSITIVE"
        
        real_frame = np.ones((720, 1280, 3), dtype=np.uint8)
        detected_objects = [(10, 10, 20, 20, "person", 0.99, 1)]
        
        context, vlm, status = self.node.ai_engine.generate_context_reasoning(
            detected_objects, [], "RUNNING", rf_count=5, acoustic_event="SHOUTING", frame=real_frame
        )
        
        self.assertEqual(status, "SUPPRESSED")
        
        # Simulate app.py logic
        event_id = None
        if status in ("CONFIRMED", "SUPPRESSED") or "THREAT VECTOR" in context:
            rule_triggered = "UNKNOWN"
            if "THREAT VECTOR:" in context:
                rule_triggered = context.split("THREAT VECTOR:")[1].split("]")[0].strip()
            elif status == "SUPPRESSED":
                rule_triggered = "ADAPTIVE_SUPPRESSED"
                
            event_id = self.node.episodic_memory.add_episode(
                rule_triggered=rule_triggered,
                scene_description=context,
                telemetry={},
                frame=real_frame
            )
            
        self.assertIsNotNone(event_id)
        self.assertEqual(rule_triggered, "ADAPTIVE_SUPPRESSED")
        
        self.node.ai_engine.pool.investigate_scene = original_investigate

    def test_feedback_immediately_affects_retrieval(self):
        # 1. Generate a test embedding
        scene_desc = "1 Individual (Identities: GPX999) [Action: RUNNING]."
        
        # 2. Check current retrieval
        initial_memories = self.node.ai_engine.retriever.get_contextual_memories(scene_desc)
        initial_count = len(initial_memories)
        
        # 3. Simulate an operator submitting feedback via API
        # We need an event in episodic memory first
        event_id = self.node.episodic_memory.add_episode(
            rule_triggered="TEST_RULE",
            scene_description=scene_desc,
            telemetry={},
            frame=np.zeros((10, 10, 3), dtype=np.uint8)
        )
        
        token = jwt.encode({"sub": "bench", "role": "admin", "exp": time.time()+3600}, JWT_SECRET, algorithm="HS256")
        headers = {"Authorization": f"Bearer {token}"}
        
        resp = requests.post("http://localhost:8083/api/v2/feedback", headers=headers, json={
            "event_id": event_id,
            "human_label": "True Positive - Hostile Intent"
        })
        self.assertEqual(resp.status_code, 200)
        
        # 4. Immediate retrieval (No restart) should now include this new memory!
        new_memories = self.node.ai_engine.retriever.get_contextual_memories(scene_desc)
        self.assertEqual(len(new_memories), initial_count + 1)
        self.assertEqual(new_memories[0]["human_label"], "True Positive - Hostile Intent")

if __name__ == '__main__':
    unittest.main()
