import unittest
import time
import json
import jwt
import threading
import os
import requests
from fluxstate_security.app import FluxStateNode

JWT_SECRET = os.environ.get("FLUX_JWT_SECRET", "flux_edge_secret_key_992")

class TestFeedbackAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We start the API server on a separate port for tests
        os.environ["ADAPTIVE_INTELLIGENCE"] = "1"
        cls.node = FluxStateNode(stream_source="") # Mock stream
        cls.node.start_api_server(port=8081)
        time.sleep(1) # Let server boot

    @classmethod
    def tearDownClass(cls):
        cls.node.stop()
        
    def generate_token(self, role="admin", exp_offset=3600):
        payload = {
            "sub": "user123",
            "role": role,
            "exp": time.time() + exp_offset
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

    def test_missing_auth(self):
        resp = requests.post("http://localhost:8081/api/v2/feedback", json={})
        self.assertEqual(resp.status_code, 401)
        
    def test_invalid_role(self):
        token = self.generate_token(role="guest")
        resp = requests.post(
            "http://localhost:8081/api/v2/feedback", 
            headers={"Authorization": f"Bearer {token}"},
            json={}
        )
        self.assertEqual(resp.status_code, 401)
        
    def test_missing_event_id(self):
        token = self.generate_token()
        resp = requests.post(
            "http://localhost:8081/api/v2/feedback", 
            headers={"Authorization": f"Bearer {token}"},
            json={"human_label": "False Positive"}
        )
        self.assertEqual(resp.status_code, 400)
        
    def test_expired_event(self):
        token = self.generate_token()
        resp = requests.post(
            "http://localhost:8081/api/v2/feedback", 
            headers={"Authorization": f"Bearer {token}"},
            json={"event_id": "nonexistent_event", "human_label": "False Positive"}
        )
        self.assertEqual(resp.status_code, 404)

    def test_successful_feedback(self):
        # 1. Create a dummy event
        event_id = self.node.episodic_memory.add_episode(
            rule_triggered="TEST_RULE",
            scene_description="Test scene with anomalous entity.",
            telemetry={}
        )
        
        # 2. Submit feedback
        token = self.generate_token()
        resp = requests.post(
            "http://localhost:8081/api/v2/feedback", 
            headers={"Authorization": f"Bearer {token}"},
            json={"event_id": event_id, "human_label": "True Positive"}
        )
        self.assertEqual(resp.status_code, 200)
        
        # 3. Verify it was embedded and stored in ForensicDatabase
        results = self.node.forensics.query_forensics("Test scene")
        # Just ensure it didn't crash; query_forensics only checks `events` table right now, 
        # but the feedback goes to `semantic_ledger`.
        
        # We can check the vector cache directly
        with self.node.forensics._cache_lock:
            self.assertIn(event_id, self.node.forensics.id_cache)

    def test_concurrent_feedback(self):
        # Add 10 events
        events = [self.node.episodic_memory.add_episode("RULE", "Desc", {}) for _ in range(10)]
        token = self.generate_token()
        
        results = []
        def worker(ev_id):
            r = requests.post(
                "http://localhost:8081/api/v2/feedback", 
                headers={"Authorization": f"Bearer {token}"},
                json={"event_id": ev_id, "human_label": "Valid"}
            )
            results.append(r.status_code)
            
        threads = [threading.Thread(target=worker, args=(ev,)) for ev in events]
        for t in threads: t.start()
        for t in threads: t.join()
        
        self.assertTrue(all(c == 200 for c in results))

def run_benchmark():
    print("\n--- Running API Integration Benchmarks ---")
    os.environ["ADAPTIVE_INTELLIGENCE"] = "1"
    node = FluxStateNode(stream_source="") 
    node.start_api_server(port=8082)
    time.sleep(1)
    
    token = jwt.encode({"sub": "bench", "role": "admin", "exp": time.time()+3600}, JWT_SECRET, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    
    # Pre-populate 100 events
    events = [node.episodic_memory.add_episode("RULE", "Desc", {}) for _ in range(100)]
    
    # 1. API Latency Benchmark (Includes Embedding + DB Write)
    t0 = time.time()
    for ev in events[:20]: # Run 20 synchronous feedback loops
        requests.post("http://localhost:8082/api/v2/feedback", headers=headers, json={"event_id": ev, "human_label": "False Positive"})
    t1 = time.time()
    avg_latency = ((t1 - t0) / 20) * 1000
    print(f"Average /feedback API Latency (Sync): {avg_latency:.2f} ms")
    
    # 2. Concurrent Performance
    t0 = time.time()
    def worker_bench(ev_id):
        requests.post("http://localhost:8082/api/v2/feedback", headers=headers, json={"event_id": ev_id, "human_label": "True Positive"})
        
    threads = [threading.Thread(target=worker_bench, args=(ev,)) for ev in events[20:]]
    for t in threads: t.start()
    for t in threads: t.join()
    t1 = time.time()
    avg_concurrent = ((t1 - t0) / 80) * 1000
    print(f"Average /feedback API Latency (80 Concurrent): {avg_concurrent:.2f} ms")
    
    node.stop()

if __name__ == '__main__':
    import sys
    if '--benchmark' in sys.argv:
        run_benchmark()
    else:
        unittest.main()
