"""
Database Logger for Manhattan Steiner Benchmarks
================================================
Handles SQLite initialization and data insertion for mass statistical analysis.
Includes support for obstacles and stochastic temperature tracking.
"""

import sqlite3
import json

def init_db(db_path="benchmark_results.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table 1: Configurations (The 'Map' setup)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS configurations (
        config_id INTEGER PRIMARY KEY AUTOINCREMENT,
        n_terminals INTEGER,
        map_seed INTEGER,
        mst_length REAL,
        mst_time REAL,
        greedy_length REAL,
        greedy_time REAL,
        terminals_json TEXT,
        obstacles_json TEXT,
        greedy_segments_json TEXT
    )
    """)
    
    # Table 2: Stochastic Trials
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trials (
        trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id INTEGER,
        trial_seed INTEGER,
        temperature REAL,
        stoch_length REAL,
        stoch_time REAL,
        is_winner BOOLEAN,
        stoch_segments_json TEXT,
        FOREIGN KEY (config_id) REFERENCES configurations (config_id)
    )
    """)
    
    conn.commit()
    return conn

def log_configuration(conn, n_terminals, map_seed, mst_l, mst_t, greedy_l, greedy_t, terminals, obstacles, greedy_segments):
    cursor = conn.cursor()
    
    # Convert obstacles to JSON-serializable list
    obs_data = [
        {"min_x": float(o.min_x), "min_y": float(o.min_y), "max_x": float(o.max_x), "max_y": float(o.max_y)}
        for o in obstacles
    ]
    
    cursor.execute("""
    INSERT INTO configurations (
        n_terminals, map_seed, mst_length, mst_time, greedy_length, greedy_time, terminals_json, obstacles_json, greedy_segments_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        n_terminals, 
        map_seed, 
        mst_l, 
        mst_t, 
        greedy_l, 
        greedy_t, 
        json.dumps(terminals.tolist()), 
        json.dumps(obs_data),
        json.dumps(greedy_segments)
    ))
    conn.commit()
    return cursor.lastrowid

def log_trial(conn, config_id, trial_seed, temperature, stoch_l, stoch_t, is_winner, stoch_segments):
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO trials (config_id, trial_seed, temperature, stoch_length, stoch_time, is_winner, stoch_segments_json)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (config_id, trial_seed, float(temperature), stoch_l, stoch_t, is_winner, json.dumps(stoch_segments)))
    conn.commit()
