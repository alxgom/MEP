"""
Optimized Progressive Mass Benchmark (v2.1)
===========================================
Pre-computes APSP once per map and runs trials in parallel.
Logs full coordinate segments for topological visualization.
"""

import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver
from db_logger import init_db, log_configuration, log_trial

OBSTACLES_DATA = [[200, 200, 300, 600], [500, 100, 600, 400], [100, 400, 400, 500]]

def run_single_trial(args):
    """Worker: Runs a stochastic trial using PRECOMPUTED data."""
    config_id, t_indices, dist_matrix, pred, n_nodes, node_map, nodes, obstacles, seed, temp, greedy_l = args
    
    solver = ObstacleSteinerSolver.from_precomputed(t_indices, dist_matrix, pred, n_nodes, node_map, nodes, obstacles)
    
    np.random.seed(seed)
    start = time.time()
    res = solver.solve_fast_corner(stochastic=True, temperature=temp)
    duration = time.time() - start
    
    # Convert segments to coordinate pairs
    stoch_segments = []
    for u, v in res["segments"]:
        stoch_segments.append([nodes[u].tolist(), nodes[v].tolist()])

    is_winner = res["weight"] < (greedy_l - 1e-6)
    return config_id, seed, temp, res["weight"], duration, is_winner, stoch_segments

def main():
    print("Initializing Optimized Progressive Benchmark with Visualization Support...")
    if os.path.exists("benchmark_results.db"): os.remove("benchmark_results.db")
    conn = init_db()
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in OBSTACLES_DATA]
    
    n_values = range(40, 105, 5)
    samples_per_n = 5
    trials_per_map = 50
    temperature = 0.1

    for n in n_values:
        print(f"\n>>> Processing N={n} block...")
        for s_idx in range(samples_per_n):
            map_seed = s_idx * 1000 + n
            np.random.seed(map_seed)
            
            # 1. Generate Valid Map
            terminals = []
            while len(terminals) < n:
                p = np.random.rand(2) * 800
                if not any(o.contains(p[0], p[1]) for o in obstacles): terminals.append(p)
            terminals = np.array(terminals)
            
            # 2. Pre-compute APSP (The heavy part - do it ONCE)
            print(f"  Map {s_idx+1}/5: Pre-computing Geodesic Matrix...")
            env = GridEnvironment(terminals, obstacles)
            
            # 3. Run Baselines
            solver = ObstacleSteinerSolver(env)
            mst_res = solver.solve_mst()
            greedy_res = solver.solve_fast_corner(stochastic=False)
            
            # Convert greedy segments to coords
            greedy_segments = []
            for u, v in greedy_res["segments"]:
                greedy_segments.append([env.nodes[u].tolist(), env.nodes[v].tolist()])

            config_id = log_configuration(conn, n, map_seed, mst_res["weight"], 0, greedy_res["weight"], 0, terminals, obstacles, greedy_segments)
            
            # 4. Dispatch Trials in Parallel
            print(f"  Dispatching {trials_per_map} parallel trials...")
            jobs = [(config_id, env.terminal_node_indices, env.dist_matrix, env.predecessors, 
                     env.n_nodes, env.node_map, env.nodes, obstacles, 
                     trial_idx + 7000, temperature, greedy_res["weight"]) 
                    for trial_idx in range(trials_per_map)]
            
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_single_trial, job) for job in jobs]
                for future in as_completed(futures):
                    cid, t_seed, temp, w, dur, win, segments = future.result()
                    log_trial(conn, cid, t_seed, temp, w, dur, win, segments)
            
            print(f"  Map {s_idx+1} complete and logged.")

    print("\nBenchmark Suite Complete!")
    conn.close()

if __name__ == "__main__":
    main()
