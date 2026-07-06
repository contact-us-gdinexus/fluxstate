import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fluxstate_security.core.adaptation.prompt_builder import PromptBuilder
from fluxstate_security.core.adaptation.retriever import SemanticRetriever
from fluxstate_security.core.forensics import ForensicDatabase
from fluxstate_security.core.agent import MLXModelPool

def _clean_db(path):
    for ext in ['', '-wal', '-shm']:
        p = path + ext
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

class TestRetrievalLayer(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_retrieval.db"
        _clean_db(self.db_path)
        self.db = ForensicDatabase(db_path=self.db_path)
        self.agent_pool = MLXModelPool(enable_adaptive=False) # Adaptive false prevents loading embedding model natively to save time/memory in tests, we'll mock it
        
        # Mocking embedder for controlled testing
        self.agent_pool.generate_embedding = lambda text: np.array([1.0 if text == "Target" else 0.0] * 384, dtype=np.float32)
        
        self.retriever = SemanticRetriever(db=self.db, agent_pool=self.agent_pool, similarity_threshold=0.85)
        self.builder = PromptBuilder(max_memories=3, min_similarity=0.85)

    def tearDown(self):
        self.db = None
        _clean_db(self.db_path)

    def test_empty_ledger(self):
        memories = self.retriever.get_contextual_memories("Target", top_k=3)
        self.assertEqual(len(memories), 0)
        
        prompt = self.builder.build_prompt("Target", "RULE_1", memories)
        self.assertIn("Initiate Visual Threat Analysis", prompt)
        self.assertNotIn("Historical Context", prompt)

    def test_similarity_threshold_behavior(self):
        # Insert a matching vector
        self.db.log_feedback_event("R1", "Match", np.array([1.0]*384, dtype=np.float32), "FP", "op", "role")
        # Insert a non-matching vector (orthogonal)
        self.db.log_feedback_event("R1", "NoMatch", np.array([0.0, 1.0] + [0.0]*382, dtype=np.float32), "TP", "op", "role")
        
        memories = self.retriever.get_contextual_memories("Target", top_k=5)
        
        # Only the matching vector should be retrieved because threshold is 0.85
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["scene_description"], "Match")

    def test_retrieval_ordering(self):
        # We manually insert vectors with varying similarity
        vec_best = np.array([1.0]*384, dtype=np.float32) # Sim 1.0 to [1.0]*384
        vec_mid = np.array([0.95]*384, dtype=np.float32) # Less similarity if not normalized identically, but wait cosine sim of [1,1,1...] and [0.95,0.95...] is 1.0!
        
        # Let's mock a deterministic dot product instead of messing with exact vectors
        self.retriever.similarity_threshold = 0.5
        
        # Create vectors that explicitly have different cosine similarities to [1.0, 0, 0, ...]
        v1 = np.zeros(384, dtype=np.float32)
        v1[0] = 1.0 # Sim = 1.0
        
        v2 = np.zeros(384, dtype=np.float32)
        v2[0] = 0.6
        v2[1] = 0.8 # Sim = 0.6
        
        v3 = np.zeros(384, dtype=np.float32)
        v3[0] = 0.8
        v3[1] = 0.6 # Sim = 0.8
        
        # We need to change the mock generate_embedding to match these
        self.agent_pool.generate_embedding = lambda text: v1
        
        self.db.log_feedback_event("R1", "Best", v1, "FP", "op", "role")
        self.db.log_feedback_event("R1", "Worst", v2, "FP", "op", "role")
        self.db.log_feedback_event("R1", "Mid", v3, "FP", "op", "role")
        
        memories = self.retriever.get_contextual_memories("Query", top_k=3)
        self.assertEqual(len(memories), 3)
        self.assertEqual(memories[0]["scene_description"], "Best")
        self.assertEqual(memories[1]["scene_description"], "Mid")
        self.assertEqual(memories[2]["scene_description"], "Worst")

    def test_prompt_formatting_and_limits(self):
        # Create 5 memories, builder should limit to 3
        mems = []
        for i in range(5):
            mems.append({
                "similarity": 0.99,
                "human_label": f"LABEL_{i}",
                "scene_description": f"DESC_{i}"
            })
            
        prompt = self.builder.build_prompt("SCENE", "RULE", mems)
        self.assertIn("DESC_0", prompt)
        self.assertIn("DESC_2", prompt)
        self.assertNotIn("DESC_3", prompt) # Enforced max_memories=3
        self.assertNotIn("DESC_4", prompt)
        
    def test_malformed_history(self):
        mems = [
            {"similarity": 0.99, "human_label": "FP", "scene_description": "Valid"},
            {"similarity": 0.99, "human_label": None, "scene_description": "Invalid Label"},
            {"similarity": 0.99, "human_label": "FP"} # Missing desc
        ]
        prompt = self.builder.build_prompt("SCENE", "RULE", mems)
        self.assertIn("Valid", prompt)
        self.assertNotIn("Invalid Label", prompt)

def run_benchmark():
    print("\\n--- Running Retrieval Layer Benchmarks ---")
    db_path = "benchmark_retrieval.db"
    _clean_db(db_path)
    
    db = ForensicDatabase(db_path=db_path)
    # Enable true adaptive pool for real embedding latency
    pool = MLXModelPool(enable_adaptive=True) 
    
    # Pre-warm pool
    _ = pool.generate_embedding("Warmup")
    
    retriever = SemanticRetriever(db=db, agent_pool=pool, similarity_threshold=0.85)
    builder = PromptBuilder()
    
    print("Inserting 1,000 vectors for benchmark scale...")
    for _ in range(1000):
        emb = np.random.rand(384).astype(np.float32)
        db.log_feedback_event("R", "D", emb, "L", "O", "R")
        
    print("Benchmarking Full Pipeline (Embed -> Retrieve -> Prompt)...")
    
    start_time = time.time()
    iters = 50
    for _ in range(iters):
        mems = retriever.get_contextual_memories("A person is walking near the fence line at night carrying a bag.")
        prompt = builder.build_prompt("A person is walking near the fence line at night carrying a bag.", "PERIMETER_BREACH", mems)
        
    total_time = time.time() - start_time
    avg_latency = (total_time / iters) * 1000
    
    print(f"Average Pipeline Latency: {avg_latency:.2f} ms")
    if avg_latency < 50:
        print("[PASS] Latency is strictly under 50ms.")
    else:
        print("[FAIL] Latency exceeds 50ms budget.")
        
    _clean_db(db_path)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        run_benchmark()
    else:
        unittest.main()
