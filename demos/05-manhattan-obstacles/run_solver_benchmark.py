"""
Full Solver Benchmark — All 8 Solvers
======================================
Sweeps terminal count N across multiple random maps with fixed obstacles.
Reports true routed length (not metric closure weight) for every solver.
Writes results to solver_benchmark.db (non-destructive, separate from benchmark_results.db).

KPIs per solver per map:
  - true_length : actual routed pipe length (sum of unique segment distances)
  - saving_pct  : % reduction vs MST baseline
  - time_ms     : wall-clock solve time in milliseconds
  - n_steiners  : number of Steiner junction points added
"""

import os
import time
import sqlite3
import numpy as np
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH         = "solver_benchmark.db"
N_VALUES        = [10, 20, 30, 40, 50]   # terminal counts to sweep
MAPS_PER_N      = 5                       # independent random maps per N
DOMAIN_SIZE     = 800                     # bounding box (pixels)
OBSTACLES_DATA  = [
    [150, 150, 280, 450],
    [400,  80, 530, 320],
    [100, 500, 380, 600],
    [500, 400, 680, 620],
]

SOLVERS = [
    ("MST",         lambda s: s.solve_mst()),
    ("KMB",         lambda s: s.solve_kou()),
    ("Greedy",      lambda s: s.solve_greedy()),
    ("FastCorner",  lambda s: s.solve_fast_corner()),
    ("Prune",       lambda s: s.solve_prune()),
    ("StochKMB",    lambda s: s.solve_stochastic_kou(n_trials=20, perturbation=0.10)),
    ("AnisoKMB",    lambda s: s.solve_anisotropic_kou(w_x=1.0, w_y=2.0, n_trials=15, sigma=0.3)),
    ("MonteCarlo",  lambda s: s.solve_monte_carlo()),
]

# ── Database ────────────────────────────────────────────────────────────────────
def init_db(path):
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE results (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            n           INTEGER,
            map_seed    INTEGER,
            n_nodes     INTEGER,
            solver      TEXT,
            true_length REAL,
            mst_length  REAL,
            saving_pct  REAL,
            time_ms     REAL,
            n_steiners  INTEGER
        )
    """)
    conn.commit()
    return conn

def log_row(conn, n, seed, n_nodes, solver_name, true_len, mst_len, time_ms, n_steiners):
    saving = (mst_len - true_len) / mst_len * 100 if mst_len > 0 else 0.0
    conn.execute(
        "INSERT INTO results (n, map_seed, n_nodes, solver, true_length, mst_length, saving_pct, time_ms, n_steiners) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (n, seed, n_nodes, solver_name, true_len, mst_len, saving, time_ms, n_steiners)
    )
    conn.commit()

# ── Map Generation ──────────────────────────────────────────────────────────────
def make_map(n, seed):
    obstacles = [Obstacle(*o) for o in OBSTACLES_DATA]
    np.random.seed(seed)
    terminals = []
    while len(terminals) < n:
        p = np.random.rand(2) * DOMAIN_SIZE
        if not any(o.contains(p[0], p[1]) for o in obstacles):
            terminals.append(p)
    return np.array(terminals), obstacles

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Full Solver Benchmark — All 8 Solvers")
    print("  True routed length (deduplicated segments) as primary metric.")
    print("=" * 72)

    conn = init_db(DB_PATH)

    for n in N_VALUES:
        print(f"\n{'─'*72}")
        print(f"  N = {n} terminals  ({MAPS_PER_N} maps)")
        print(f"{'─'*72}")

        # Accumulators for summary row
        totals = {name: {"len": 0.0, "sav": 0.0, "ms": 0.0, "st": 0} for name, _ in SOLVERS}

        for map_idx in range(MAPS_PER_N):
            seed = n * 1000 + map_idx
            terminals, obstacles = make_map(n, seed)

            # Build environment once per map
            t0 = time.time()
            env = GridEnvironment(terminals, obstacles)
            build_ms = (time.time() - t0) * 1000

            mst_len = None  # filled on first solver (MST)

            for solver_name, solver_fn in SOLVERS:
                s = ObstacleSteinerSolver(env)
                t0 = time.time()
                res = solver_fn(s)
                elapsed_ms = (time.time() - t0) * 1000

                true_len = res["weight"]
                if mst_len is None:
                    mst_len = true_len   # first solver is always MST

                n_steiners = len(res["steiner_indices"])
                log_row(conn, n, seed, env.n_nodes, solver_name,
                        true_len, mst_len, elapsed_ms, n_steiners)

                totals[solver_name]["len"] += true_len
                totals[solver_name]["sav"] += (mst_len - true_len) / mst_len * 100
                totals[solver_name]["ms"]  += elapsed_ms
                totals[solver_name]["st"]  += n_steiners

            print(f"  Map {map_idx+1}/{MAPS_PER_N}  seed={seed}  nodes={env.n_nodes}  build={build_ms:.0f}ms  mst={mst_len:.1f}")

        # Print summary table for this N
        print(f"\n  {'Solver':<14} {'Avg Length':>12}  {'Avg Saving':>11}  {'Avg Time':>10}  {'Avg Steiners':>13}")
        print(f"  {'─'*14} {'─'*12}  {'─'*11}  {'─'*10}  {'─'*13}")
        mst_avg = totals["MST"]["len"] / MAPS_PER_N
        for solver_name, _ in SOLVERS:
            t = totals[solver_name]
            avg_len = t["len"] / MAPS_PER_N
            avg_sav = t["sav"] / MAPS_PER_N
            avg_ms  = t["ms"]  / MAPS_PER_N
            avg_st  = t["st"]  / MAPS_PER_N
            marker = " *" if avg_len == min(totals[s]["len"] / MAPS_PER_N for s, _ in SOLVERS) else ""
            print(f"  {solver_name:<14} {avg_len:>12.1f}  {avg_sav:>10.1f}%  {avg_ms:>9.1f}ms  {avg_st:>12.1f}{marker}")

    print(f"\n{'='*72}")
    print(f"  Results saved → {DB_PATH}")
    print(f"{'='*72}")
    conn.close()

if __name__ == "__main__":
    main()
