"""
Full Solver Benchmark v2 — Constant Density + Fixed Steiner Counts
====================================================================
KEY FIX from v1:
  - Domain scales with sqrt(N) so terminal DENSITY is constant across all N.
  - Obstacle coordinates scale proportionally, preserving coverage fraction.
  - Terminals that fall inside obstacles are re-sampled (not just rejected).
  - Steiner count now reflects real junctions only (two-stage pruning).

Density model:
  Base: N_REF=10 terminals in BASE_DOMAIN x BASE_DOMAIN area.
  For any N: domain_side = BASE_DOMAIN * sqrt(N / N_REF)
             obstacles   = BASE_OBSTACLES * scale_factor

KPIs per solver per map:
  true_length : actual routed pipe length (deduplicated segments, coordinate units)
  saving_pct  : % reduction vs MST baseline
  time_ms     : wall-clock solve time
  n_steiners  : real junctions (degree >= 3), not pass-throughs
"""

import os
import time
import sqlite3
import numpy as np
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH    = "solver_benchmark_v2.db"
N_REF      = 10       # reference terminal count for density calibration
BASE_W     = 1000     # room width  at N=N_REF  (aspect ratio preserved at all N)
BASE_H     = 700      # room height at N=N_REF
N_VALUES   = [10, 20, 30, 40, 50]
MAPS_PER_N = 5

# Obstacles as FRACTIONS of (W, H) — shape-agnostic, always covers the same
# proportion of the room regardless of room dimensions or scale factor.
#   [x1/W, y1/H, x2/W, y2/H]
OBSTACLE_FRACS = [
    [0.10, 0.15,  0.28, 0.55],   # left-center block
    [0.42, 0.08,  0.58, 0.40],   # top-right block
    [0.08, 0.65,  0.42, 0.82],   # bottom-left strip
    [0.60, 0.52,  0.82, 0.80],   # bottom-right block
]

SOLVERS = [
    ("MST",            lambda s: s.solve_mst()),
    ("KMB",            lambda s: s.solve_kou()),
    ("Greedy",         lambda s: s.solve_greedy()),
    ("FastCorner",     lambda s: s.solve_fast_corner()),
    ("Prune",          lambda s: s.solve_prune()),
    ("StochKMB",       lambda s: s.solve_stochastic_kou(n_trials=20, perturbation=0.10)),
    ("Aniso_Isotropic",lambda s: s.solve_anisotropic_kou(w_x=1.0, w_y=1.0, n_trials=1, sigma=0.0)),
    ("Aniso_Asymmetric",lambda s: s.solve_anisotropic_kou(w_x=1.0, w_y=2.0, n_trials=1, sigma=0.0)),
    ("Aniso_Stochastic",lambda s: s.solve_anisotropic_kou(w_x=1.5, w_y=1.5, n_trials=15, sigma=0.3)),
    ("MonteCarlo",     lambda s: s.solve_monte_carlo()),
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
            room_w      REAL,
            room_h      REAL,
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

def log_row(conn, n, seed, W, H, n_nodes, name, true_len, mst_len, ms, n_st):
    saving = (mst_len - true_len) / mst_len * 100 if mst_len > 0 else 0.0
    conn.execute(
        "INSERT INTO results VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
        (n, seed, W, H, n_nodes, name, true_len, mst_len, saving, ms, n_st)
    )
    conn.commit()

# ── Map Generation — Constant Density, Preserved Room Shape ────────────────────
def make_scaled_map(n, seed):
    """
    Scale room uniformly by sqrt(N/N_REF) — maintains aspect ratio and
    obstacle coverage fraction at all N:

        scale = sqrt(N / N_REF)
        W'    = BASE_W * scale
        H'    = BASE_H * scale   (same factor → same W:H ratio)
        Area' = W'*H' = BASE_W*BASE_H * (N/N_REF)   → density is constant

    Obstacles are defined as fractions of (W, H), so they automatically
    scale with the room and maintain their relative positions and sizes.

    Terminals inside obstacles are rejected and re-sampled.
    """
    scale = np.sqrt(n / N_REF)
    W = BASE_W * scale
    H = BASE_H * scale

    obstacles = [
        Obstacle(f[0]*W, f[1]*H, f[2]*W, f[3]*H)
        for f in OBSTACLE_FRACS
    ]

    rng = np.random.default_rng(seed)
    terminals = []
    while len(terminals) < n:
        xs = rng.random(n * 4) * W
        ys = rng.random(n * 4) * H
        for x, y in zip(xs, ys):
            if not any(ob.contains(x, y) for ob in obstacles):
                terminals.append([x, y])
                if len(terminals) == n:
                    break

    return np.array(terminals), obstacles, W, H

# ── Helpers ─────────────────────────────────────────────────────────────────────
def allowed_area(W, H, obstacles):
    """Routing area = room area minus obstacle footprints."""
    obs_area = sum((o.max_x - o.min_x) * (o.max_y - o.min_y) for o in obstacles)
    return W * H - obs_area

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 76)
    print("  Full Solver Benchmark v2 -- Constant Density, Preserved Room Shape")
    print(f"  Base room: {BASE_W}x{BASE_H} at N={N_REF}. Both dims scale by sqrt(N/{N_REF}).")
    print(f"  Obstacles: fractional coords -> same coverage fraction at all N.")
    print("=" * 76)

    conn = init_db(DB_PATH)

    for n in N_VALUES:
        _, obs_ref, W_ref, H_ref = make_scaled_map(n, 0)
        density = n / allowed_area(W_ref, H_ref, obs_ref)

        print(f"\n{'─'*76}")
        print(f"  N={n}  room={W_ref:.0f}x{H_ref:.0f}  density={density*1e4:.2f} terminals/10kpx2")
        print(f"{'─'*76}")

        totals = {name: {"len": 0.0, "sav": 0.0, "ms": 0.0, "st": 0} for name, _ in SOLVERS}

        for map_idx in range(MAPS_PER_N):
            seed = n * 1000 + map_idx
            terminals, obstacles, W, H = make_scaled_map(n, seed)

            t0 = time.time()
            env = GridEnvironment(terminals, obstacles)
            build_ms = (time.time() - t0) * 1000

            mst_len = None

            for solver_name, solver_fn in SOLVERS:
                s = ObstacleSteinerSolver(env)
                t0 = time.time()
                res = solver_fn(s)
                elapsed_ms = (time.time() - t0) * 1000

                true_len   = res["weight"]
                n_steiners = len(res["steiner_indices"])

                if mst_len is None:
                    mst_len = true_len

                log_row(conn, n, seed, W, H, env.n_nodes, solver_name,
                        true_len, mst_len, elapsed_ms, n_steiners)

                totals[solver_name]["len"] += true_len
                totals[solver_name]["sav"] += (mst_len - true_len) / mst_len * 100
                totals[solver_name]["ms"]  += elapsed_ms
                totals[solver_name]["st"]  += n_steiners

            print(f"  Map {map_idx+1}/{MAPS_PER_N}  seed={seed}  nodes={env.n_nodes}"
                  f"  build={build_ms:.0f}ms  mst={mst_len:.1f}")

        # Summary table
        print(f"\n  {'Solver':<14} {'Avg Length':>12}  {'Saving':>8}  {'Time':>10}  {'Steiners':>9}")
        print(f"  {'─'*14} {'─'*12}  {'─'*8}  {'─'*10}  {'─'*9}")
        best_len = min(totals[s]["len"] / MAPS_PER_N for s, _ in SOLVERS)
        for solver_name, _ in SOLVERS:
            t   = totals[solver_name]
            aL  = t["len"] / MAPS_PER_N
            aS  = t["sav"] / MAPS_PER_N
            aMs = t["ms"]  / MAPS_PER_N
            aSt = t["st"]  / MAPS_PER_N
            mark = " *" if np.isclose(aL, best_len) else ""
            print(f"  {solver_name:<14} {aL:>12.1f}  {aS:>7.1f}%  {aMs:>9.1f}ms  {aSt:>8.1f}{mark}")

    print(f"\n{'='*76}")
    print(f"  Results saved -> {DB_PATH}")
    print(f"{'='*76}")
    conn.close()

if __name__ == "__main__":
    main()
