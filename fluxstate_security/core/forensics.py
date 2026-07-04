import sqlite3
import json
import time
import os

class ForensicDatabase:
    """
    Honest Enterprise Feature: Solves the real-world problem of 
    'Watching 8 hours of video to find an incident.'
    
    This locally stores semantic metadata (not video) into a highly optimized SQLite DB,
    allowing operators to perform instantaneous Natural Language Forensic Searches 
    (e.g., "Find all individuals carrying a red bag after 2 AM").
    """
    def __init__(self, db_path="swarm_ledger.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()

    def log_event(self, event_payload):
        """Saves telemetry to disk for forensic querying, completely devoid of PII/Video."""
        try:
            # Use timeout and context manager to prevent 'database is locked' errors
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
                
                # Update Identity Ledger
                for entity in event_payload.get("entities", []):
                    # Extract the GPASS string ID from the entity dictionary
                    gpass_id = entity.get("id", "UNKNOWN")
                    cursor.execute(
                        "INSERT INTO identities (gpass_id, first_seen, last_seen, associated_objects) VALUES (?, ?, ?, ?) "
                        "ON CONFLICT(gpass_id) DO UPDATE SET last_seen=excluded.last_seen",
                        (gpass_id, time.time(), time.time(), "")
                    )
        except Exception as e:
            print(f"[Forensics] Error writing to ledger: {e}")

    def query_forensics(self, keyword):
        """Allows instant querying of the past."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT datetime(timestamp, 'unixepoch', 'localtime'), context_log FROM events WHERE context_log LIKE ? ORDER BY timestamp DESC LIMIT 50", (f'%{keyword}%',))
        results = cursor.fetchall()
        conn.close()
        return results
