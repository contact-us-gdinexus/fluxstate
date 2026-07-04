import pytest
import os
import json
import sqlite3
import time
from fluxstate_security.core.forensics import ForensicDatabase

def test_forensics_database_initialization():
    """Test that the SQLite database is created properly."""
    db_path = "test_ledger.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = ForensicDatabase(db_path=db_path)
    assert os.path.exists(db_path), "Database file was not created."
    
    # Check tables
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]
    assert "events" in tables, "Events table missing"
    assert "identities" in tables, "Identities table missing"
    
    conn.close()
    os.remove(db_path)

def test_forensics_logging():
    """Test logging an event to the SQLite database."""
    db_path = "test_ledger_log.db"
    db = ForensicDatabase(db_path=db_path)
    
    payload = {
        "context_log": "TEST ANOMALY 🚨",
        "entities": [{"id": "GPX-1234", "class": "person"}]
    }
    
    db.log_event(payload)
    
    results = db.query_forensics("TEST")
    assert len(results) == 1, "Failed to log or query forensic event"
    assert "TEST ANOMALY 🚨" in results[0][1], "Context log mismatch"
    
    os.remove(db_path)

def test_intelligence_policy_loading():
    """Test that intelligence_policy.json is valid JSON."""
    policy_path = os.path.join(os.path.dirname(__file__), "..", "fluxstate_security", "intelligence_policy.json")
    assert os.path.exists(policy_path), "Policy file missing"
    with open(policy_path, "r") as f:
        data = json.load(f)
        assert "THREAT_VECTORS" in data, "Policy missing THREAT_VECTORS"
        assert "NODE_ID" in data, "Policy missing NODE_ID"
