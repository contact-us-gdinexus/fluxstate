import sys
import os
import time
import unittest
import numpy as np
import sqlite3

# Adjust path to import fluxstate_security
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fluxstate_security.core.forensics import ForensicDatabase

def _clean_db(path):
    for ext in ['', '-wal', '-shm']:
        p = path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

class TestForensicDatabase(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_swarm_ledger.db"
        _clean_db(self.db_path)
        self.db = ForensicDatabase(db_path=self.db_path)
        
    def tearDown(self):
        # We need to make sure db object lets go of connections if any are kept open
        self.db = None
        _clean_db(self.db_path)

    def test_init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]
        self.assertIn('semantic_ledger', tables)
        self.assertIn('events', tables)
        self.assertIn('identities', tables)

    def test_log_feedback_event_and_search(self):
        # Generate dummy 384-d embedding
        embed1 = np.random.rand(384).astype(np.float32)
        event_id = self.db.log_feedback_event(
            rule_triggered="RF_PAYLOAD_ANOMALY",
            scene_description="Man walking with 4 phones.",
            embedding=embed1,
            human_label="FALSE_POSITIVE",
            operator_id="admin-123",
            operator_role="ADMIN"
        )
        self.assertIsNotNone(event_id)
        
        # Test Search (exact match should be sim=1.0)
        results = self.db.search_similar_events(embed1, top_k=1, threshold=0.99)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], event_id)
        self.assertEqual(results[0]["human_label"], "FALSE_POSITIVE")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=4)
        self.assertEqual(results[0]["rule_triggered"], "RF_PAYLOAD_ANOMALY")
        
    def test_cosine_similarity_thresholds(self):
        # Test that dissimilar vectors are rejected
        embed1 = np.array([1, 0, 0], dtype=np.float32)
        embed2 = np.array([0, 1, 0], dtype=np.float32) # orthogonal
        
        self.db.log_feedback_event("RULE_1", "Desc1", embed1, "TP", "op1", "ADMIN")
        
        # Search with orthogonal vector should yield 0 similarity, below 0.90 threshold
        results = self.db.search_similar_events(embed2, top_k=1, threshold=0.90)
        self.assertEqual(len(results), 0)

def run_benchmark():
    print("\n--- Running ContextLedger Benchmarks ---")
    db_path = "benchmark_ledger.db"
    _clean_db(db_path)
        
    db = ForensicDatabase(db_path=db_path)
    
    # 1. Insert 10,000 vectors
    print("Inserting 10,000 feedback events (Simulating 1 year of edge data)...")
    start_time = time.time()
    for _ in range(10000):
        emb = np.random.rand(384).astype(np.float32)
        db.log_feedback_event("BENCHMARK", "Test", emb, "FP", "op", "GUARD")
    insert_time = time.time() - start_time
    print(f"Insertion Time: {insert_time:.2f} seconds ({(insert_time/10000)*1000:.2f} ms/event)")
    
    # 2. Search Performance
    print("Benchmarking cosine similarity search across 10,000 vectors...")
    query_vec = np.random.rand(384).astype(np.float32)
    start_time = time.time()
    for _ in range(100):
        _ = db.search_similar_events(query_vec, top_k=5, threshold=0.0)
    search_time = time.time() - start_time
    print(f"Average Search Time: {(search_time/100)*1000:.2f} ms")
    
    _clean_db(db_path)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        run_benchmark()
    else:
        unittest.main()
