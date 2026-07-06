import unittest
import time
import os
import shutil
import numpy as np
import threading
from fluxstate_security.core.state_manager import EpisodicMemoryBuffer

class TestEpisodicMemoryBuffer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = "/tmp/flux_test_episodic"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.buffer = EpisodicMemoryBuffer(ttl_seconds=1, max_size=5, temp_frame_dir=self.temp_dir)

    def tearDown(self):
        self.buffer.clear()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_add_and_retrieve_episode(self):
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        event_id = self.buffer.add_episode(
            rule_triggered="TEST_RULE",
            scene_description="A test scene.",
            telemetry={"rf_count": 2},
            frame=frame
        )
        self.assertIsNotNone(event_id)
        
        ep = self.buffer.get_episode(event_id)
        self.assertIsNotNone(ep)
        self.assertEqual(ep["rule_triggered"], "TEST_RULE")
        self.assertEqual(ep["scene_description"], "A test scene.")
        self.assertEqual(ep["telemetry"]["rf_count"], 2)
        
        # Verify frame saved
        self.assertIsNotNone(ep["frame_path"])
        self.assertTrue(os.path.exists(ep["frame_path"]))

    def test_ttl_eviction(self):
        # Buffer has ttl of 1 second
        event_id = self.buffer.add_episode("RULE", "Desc", {})
        self.assertEqual(self.buffer.get_size(), 1)
        
        time.sleep(1.1)
        
        # Adding a new episode should trigger eviction of the old one
        self.buffer.add_episode("RULE2", "Desc2", {})
        self.assertEqual(self.buffer.get_size(), 1)
        
        # Old event should be gone
        self.assertIsNone(self.buffer.get_episode(event_id))

    def test_max_size_eviction(self):
        # Buffer has max_size of 5
        ids = []
        for i in range(6):
            ids.append(self.buffer.add_episode(f"RULE_{i}", f"Desc_{i}", {}))
            
        self.assertEqual(self.buffer.get_size(), 5)
        # The oldest (0th) should be evicted
        self.assertIsNone(self.buffer.get_episode(ids[0]))
        # The newest should be present
        self.assertIsNotNone(self.buffer.get_episode(ids[-1]))

    def test_concurrency(self):
        """Test concurrent appends from multiple threads."""
        buffer = EpisodicMemoryBuffer(ttl_seconds=10, max_size=1000, temp_frame_dir=self.temp_dir)
        
        def worker():
            for i in range(100):
                buffer.add_episode(f"RULE", f"Desc_{i}", {})
                
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
            
        self.assertEqual(buffer.get_size(), 1000)

    def test_memory_cleanup(self):
        """Test that files are actually deleted from disk on eviction."""
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        event_id = self.buffer.add_episode("RULE", "Desc", {}, frame=frame)
        ep = self.buffer.get_episode(event_id)
        path = ep["frame_path"]
        
        self.assertTrue(os.path.exists(path))
        
        # Force eviction by adding 5 more
        for i in range(5):
            self.buffer.add_episode("RULE", "Desc", {}, frame=frame)
            
        # The first event's file should be deleted
        self.assertFalse(os.path.exists(path))

def run_benchmark():
    print("\n--- Running Episodic Memory Benchmarks ---")
    temp_dir = "/tmp/flux_bench_episodic"
    os.makedirs(temp_dir, exist_ok=True)
    buffer = EpisodicMemoryBuffer(ttl_seconds=600, max_size=10000, temp_frame_dir=temp_dir)
    
    # 1. Append Latency (No Frame)
    t0 = time.time()
    for i in range(1000):
        buffer.add_episode("RULE", "Desc", {})
    t1 = time.time()
    avg_append = ((t1 - t0) / 1000) * 1000
    print(f"Average Append Latency (No Frame): {avg_append:.4f} ms")
    
    # 2. Append Latency (With Frame)
    frame = np.zeros((1920, 1080, 3), dtype=np.uint8) # 1080p dummy frame
    t0 = time.time()
    for i in range(50):
        buffer.add_episode("RULE", "Desc", {}, frame=frame)
    t1 = time.time()
    avg_append_frame = ((t1 - t0) / 50) * 1000
    print(f"Average Append Latency (With 1080p Frame): {avg_append_frame:.4f} ms")
    
    # 3. Lookup Latency
    last_id = buffer.buffer[-1]["event_id"]
    t0 = time.time()
    for i in range(1000):
        buffer.get_episode(last_id)
    t1 = time.time()
    avg_lookup = ((t1 - t0) / 1000) * 1000
    print(f"Average Lookup Latency: {avg_lookup:.4f} ms")
    
    # 4. RAM Usage (Object Size)
    import sys
    size_bytes = sys.getsizeof(buffer.buffer)
    for ep in buffer.buffer:
        size_bytes += sys.getsizeof(ep)
    print(f"RAM Usage (1000+ items): {size_bytes / 1024:.2f} KB")
    
    # 5. Eviction Performance
    buffer.max_size = 100
    t0 = time.time()
    buffer.add_episode("RULE", "Desc", {})
    t1 = time.time()
    print(f"Eviction of 900+ items latency: {(t1 - t0) * 1000:.2f} ms")
    
    buffer.clear()
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    import sys
    if '--benchmark' in sys.argv:
        run_benchmark()
    else:
        unittest.main()
