"""
Database Logger for Multi-Net Routing (v1.0)
===========================================
Handles SQLite persistence for routing scenarios and solver results.
Supports incremental benchmarking and decoupled plotting.
"""

import sqlite3
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

class RoutingDB:
    def __init__(self, db_path: str = "routing_results.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_topology(self):
        """Standard MEP table init."""
        cursor = self.conn.cursor()
        
        # Scenarios: Stores terminal locations for each net
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                nets_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Results: Stores metrics and path segments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id INTEGER,
                solver_name TEXT,
                total_weight REAL,
                issues INTEGER,
                segments_json TEXT,
                failed BOOLEAN,
                execution_time_ms REAL,
                FOREIGN KEY(scenario_id) REFERENCES scenarios(id),
                UNIQUE(scenario_id, solver_name)
            )
        """)
        self.conn.commit()

    def _init_tables(self):
        self._init_topology()

    def save_scenario(self, name: str, nets: Dict[str, np.ndarray]) -> int:
        """Saves a scenario and returns its ID."""
        # Convert numpy arrays to lists for JSON serialization
        nets_serializable = {k: v.tolist() for k, v in nets.items()}
        nets_json = json.dumps(nets_serializable)
        
        cursor = self.conn.cursor()
        try:
            cursor.execute("INSERT INTO scenarios (name, nets_json) VALUES (?, ?)", (name, nets_json))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            cursor.execute("SELECT id FROM scenarios WHERE name = ?", (name,))
            return cursor.fetchone()[0]

    def get_scenario(self, scenario_id: int) -> Tuple[str, Dict[str, np.ndarray]]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, nets_json FROM scenarios WHERE id = ?", (scenario_id,))
        row = cursor.fetchone()
        if not row: return None
        
        name, nets_json = row
        nets_list = json.loads(nets_json)
        nets = {k: np.array(v) for k, v in nets_list.items()}
        return name, nets

    def save_result(self, scenario_id: int, solver_name: str, result: Dict[str, Any], weight: float, issues: int):
        """Saves a specific solver's result for a scenario."""
        # result is expected to be the Dict of {net_name: {weight, segments, failed}}
        segments_json = json.dumps(result)
        failed = any(data.get("failed", False) for data in result.values())
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO results 
            (scenario_id, solver_name, total_weight, issues, segments_json, failed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (scenario_id, solver_name, weight, issues, segments_json, failed))
        self.conn.commit()

    def get_results_for_scenario(self, scenario_id: int) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT solver_name, total_weight, issues, segments_json, failed 
            FROM results WHERE scenario_id = ?
        """, (scenario_id,))
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "solver_name": row[0],
                "total_weight": row[1],
                "issues": row[2],
                "segments": json.loads(row[3]),
                "failed": bool(row[4])
            })
        return results

    def check_result_exists(self, scenario_id: int, solver_name: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM results WHERE scenario_id = ? AND solver_name = ?", (scenario_id, solver_name))
        return cursor.fetchone() is not None

    def close(self):
        self.conn.close()
