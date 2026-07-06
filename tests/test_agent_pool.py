import sys
import os
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fluxstate_security.core.agent import MLXModelPool

class TestMLXModelPool(unittest.TestCase):
    def test_singleton_pattern(self):
        pool1 = MLXModelPool(enable_adaptive=False)
        pool2 = MLXModelPool(enable_adaptive=False)
        self.assertIs(pool1, pool2, "MLXModelPool must be a singleton")

    def test_gpu_lock_initialization(self):
        pool = MLXModelPool(enable_adaptive=False)
        self.assertIsNotNone(pool._gpu_lock)

    def test_embedding_fallback(self):
        # Even if embedding model fails, it should return a 384-d float32 array
        pool = MLXModelPool(enable_adaptive=False)
        emb = pool.generate_embedding("Test")
        self.assertEqual(emb.shape, (384,))
        self.assertEqual(emb.dtype, np.float32)

def run_benchmark():
    print("\n--- Running MLXModelPool Benchmarks ---")
    start_init = time.time()
    pool = MLXModelPool(enable_adaptive=True)
    print(f"Pool Initialization Time: {time.time() - start_init:.2f} seconds")
    
    print("Benchmarking Embedding Generation (10 iterations)...")
    start_emb = time.time()
    for _ in range(10):
        _ = pool.generate_embedding("Suspicious individual near the perimeter fence.")
    total_emb = time.time() - start_emb
    print(f"Average Embedding Latency: {(total_emb / 10) * 1000:.2f} ms")

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        run_benchmark()
    else:
        unittest.main()
