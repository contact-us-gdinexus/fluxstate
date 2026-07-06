import sqlite3
import json
import time
import os
import uuid
import threading
import numpy as np

class ForensicDatabase:
    """
    Honest Enterprise Feature: Solves the real-world problem of 
    'Watching 8 hours of video to find an incident.'
    
    This locally stores semantic metadata (not video) into a highly optimized SQLite DB.
    V2: Integrates Semantic Context Ledger using numpy float32 BLOBs for fast in-memory cosine similarity.
    """
    def __init__(self, db_path="swarm_ledger.db", max_cache_size=50000):
        self.db_path = db_path
        self.max_cache_size = max_cache_size
        self._cache_lock = threading.Lock()
        
        # In-memory FAISS/Numpy flat index cache
        self.vector_cache = None
        self.id_cache = []
        self.label_cache = []
        self.timestamp_cache = []
        
        self._init_db()
        self._load_vector_index()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            # Enable WAL mode for concurrent read/write and high performance
            conn.execute('PRAGMA journal_mode = WAL;')
            conn.execute('PRAGMA synchronous = NORMAL;')
            
            cursor = conn.cursor()
            # Events table: stores temporal summaries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    event_type TEXT,
                    entities TEXT,
                    context_log TEXT
                )
            ''')
            # Identities table: tracks when specific GPASS hashes were last seen
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS identities (
                    gpass_id TEXT PRIMARY KEY,
                    first_seen REAL,
                    last_seen REAL,
                    associated_objects TEXT
                )
            ''')
            # V2: Semantic Ledger Table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS semantic_ledger (
                    id TEXT PRIMARY KEY,
                    timestamp REAL,
                    rule_triggered TEXT,
                    scene_description TEXT,
                    embedding BLOB,
                    human_label TEXT,
                    operator_id TEXT,
                    operator_role TEXT
                )
            ''')
            # Index for TTL pruning
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_semantic_timestamp ON semantic_ledger(timestamp);')
            conn.commit()

    def _load_vector_index(self):
        """Loads the semantic ledger into memory as a flat Numpy index for rapid O(N) lookup."""
        with self._cache_lock:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                cursor = conn.cursor()
                # Only load the last 90 days to prevent infinite RAM creep (TTL simulation)
                cutoff = time.time() - (90 * 24 * 3600)
                
                # We enforce max_cache_size on load as well
                cursor.execute(
                    "SELECT id, embedding, human_label, timestamp FROM semantic_ledger WHERE timestamp > ? ORDER BY timestamp DESC LIMIT ?", 
                    (cutoff, self.max_cache_size)
                )
                rows = cursor.fetchall()
                
                if not rows:
                    self.vector_cache = np.empty((0, 0), dtype=np.float32)
                    self.id_cache = []
                    self.label_cache = []
                    self.timestamp_cache = []
                    return
                
                # Rows are loaded descending (newest first). Let's reverse them so older is first (FIFO appending logic).
                rows.reverse()
                
                vectors = []
                ids = []
                labels = []
                timestamps = []
                
                for r_id, blob, label, ts in rows:
                    # Convert SQLite BLOB back to float32 numpy array
                    vec = np.frombuffer(blob, dtype=np.float32)
                    vectors.append(vec)
                    ids.append(r_id)
                    labels.append(label)
                    timestamps.append(ts)
                    
                self.vector_cache = np.vstack(vectors)
                self.id_cache = ids
                self.label_cache = labels
                self.timestamp_cache = timestamps

    def _evict_cache_if_needed(self):
        """Must be called holding self._cache_lock. Truncates memory cache if it exceeds max size."""
        if self.vector_cache is not None and self.vector_cache.shape[0] > self.max_cache_size:
            # We remove the oldest 10% to prevent continuous re-allocation thrashing
            evict_count = int(self.max_cache_size * 0.1)
            
            self.vector_cache = self.vector_cache[evict_count:, :]
            self.id_cache = self.id_cache[evict_count:]
            self.label_cache = self.label_cache[evict_count:]
            self.timestamp_cache = self.timestamp_cache[evict_count:]

    def log_event(self, event_payload):
        """Saves telemetry to disk for forensic querying, completely devoid of PII/Video."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO events (timestamp, event_type, entities, context_log) VALUES (?, ?, ?, ?)",
                    (
                        time.time(),
                        "THREAT_ESCALATION" if "🚨" in event_payload.get("context_log", "") else "OBSERVATION",
                        json.dumps(event_payload.get("entities", [])),
                        event_payload.get("context_log", "")
                    )
                )
                
                for entity in event_payload.get("entities", []):
                    gpass_id = entity.get("id", "UNKNOWN")
                    cursor.execute(
                        "INSERT INTO identities (gpass_id, first_seen, last_seen, associated_objects) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(gpass_id) DO UPDATE SET last_seen=excluded.last_seen",
                        (gpass_id, time.time(), time.time(), "")
                    )
        except Exception as e:
            print(f"[Forensics] Error writing to events ledger: {e}")

    def log_feedback_event(self, rule_triggered: str, scene_description: str, embedding: np.ndarray, human_label: str, operator_id: str, operator_role: str, event_id: str = None):
        """Stores operator feedback into the semantic ledger (SQLite + RAM Cache)."""
        event_id = event_id or str(uuid.uuid4())
        timestamp = time.time()
        
        # Ensure embedding is float32 and normalize it for cosine similarity
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        blob = embedding.tobytes()
        
        # 1. Write to DB First (Durability)
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO semantic_ledger (id, timestamp, rule_triggered, scene_description, embedding, human_label, operator_id, operator_role) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (event_id, timestamp, rule_triggered, scene_description, blob, human_label, operator_id, operator_role)
                )
        except Exception as e:
            print(f"[Forensics] Error writing semantic ledger: {e}")
            return None
            
        # 2. Update RAM Cache (Thread-Safe)
        with self._cache_lock:
            try:
                if self.vector_cache is None or self.vector_cache.shape[0] == 0:
                    self.vector_cache = embedding.reshape(1, -1)
                else:
                    self.vector_cache = np.vstack([self.vector_cache, embedding])
                    
                self.id_cache.append(event_id)
                self.label_cache.append(human_label)
                self.timestamp_cache.append(timestamp)
                
                # Check bounds
                self._evict_cache_if_needed()
            except Exception as e:
                print(f"[Forensics] Error updating semantic cache: {e}")
                
        return event_id

    def search_similar_events(self, query_embedding: np.ndarray, top_k: int = 3, threshold: float = 0.90):
        """Finds top_k similar events using fast Numpy cosine similarity."""
        with self._cache_lock:
            # We copy the references to prevent race conditions during the dot product if a vstack occurs
            v_cache = self.vector_cache
            i_cache = list(self.id_cache)
            l_cache = list(self.label_cache)
            
        if v_cache is None or v_cache.shape[0] == 0:
            return []
            
        # Normalize query
        query_embedding = query_embedding.astype(np.float32)
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
            
        # Calculate cosine similarity (dot product of normalized vectors)
        similarities = np.dot(v_cache, query_embedding)
        
        # Get indices of top_k (handle case where N < top_k)
        actual_k = min(top_k, similarities.shape[0])
        if actual_k == 0:
            return []
            
        # argpartition is faster than argsort for large arrays (O(N) vs O(N log N))
        if similarities.shape[0] > actual_k:
            top_indices = np.argpartition(similarities, -actual_k)[-actual_k:]
            # argpartition doesn't guarantee sorted order within the top-k, so we sort them
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({
                    "id": i_cache[idx],
                    "human_label": l_cache[idx],
                    "similarity": sim
                })
                
        # Fetch full metadata from DB for the results
        if results:
            try:
                with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                    cursor = conn.cursor()
                    placeholders = ','.join('?' for _ in results)
                    ids_to_fetch = [r['id'] for r in results]
                    cursor.execute(f"SELECT id, rule_triggered, scene_description FROM semantic_ledger WHERE id IN ({placeholders})", ids_to_fetch)
                    meta_rows = {row[0]: {"rule": row[1], "desc": row[2]} for row in cursor.fetchall()}
                    
                    for r in results:
                        meta = meta_rows.get(r['id'], {})
                        r['rule_triggered'] = meta.get('rule')
                        r['scene_description'] = meta.get('desc')
            except Exception as e:
                print(f"[Forensics] DB read error during semantic search: {e}")
                    
        return results

    def query_forensics(self, keyword):
        """Allows instant querying of the past."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT datetime(timestamp, 'unixepoch', 'localtime'), context_log FROM events WHERE context_log LIKE ? ORDER BY timestamp DESC LIMIT 50", (f'%{keyword}%',))
                results = cursor.fetchall()
            return results
        except Exception:
            return []
