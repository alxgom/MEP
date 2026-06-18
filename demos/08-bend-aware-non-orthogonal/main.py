import time
import numpy as np
import shapely.affinity
from shapely.geometry import Polygon
from environment import NonOrthogonalEnvironment
from solver import (
    BendAwareKMBSolver, TurnCleanupSolver, BendAwareFastCornerSolver,
    BendAwareDualGraphKMBSolver, BendAwareStochasticKMBSolver, BendAwarePruneSolver,
    BendAwareDualGraphGBFSSolver, BendAwareDualGraphFastCornerSolver,
    BendAwareDualGraphGBFSFastCornerSolver, StateExpandedSequentialFastCornerSolver,
    DualGraphSequentialFastCornerSolver
)

def create_rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Polygon:
    """Creates a rectangle centered at (cx, cy) rotated by angle_deg degrees."""
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return shapely.affinity.rotate(rect, angle_deg, origin='center')

def main():
    print("Initializing environment...")
    
    # 1. Initialize Winding Corridor Room
    room = Polygon([
        (0, 0), (1200, 0), (1200, 400), (800, 400), (800, 800), (1200, 800),
        (1200, 1200), (0, 1200), (0, 800), (400, 800), (400, 400), (0, 400)
    ])
    
    # 2. Place 7 rotated obstacles inside the room
    obs1 = create_rotated_rect(250.0, 200.0, 100.0, 60.0, 15.0)
    obs2 = create_rotated_rect(600.0, 200.0, 80.0, 120.0, -30.0)
    obs3 = create_rotated_rect(1000.0, 200.0, 60.0, 60.0, 45.0)
    obs4 = create_rotated_rect(1000.0, 1000.0, 120.0, 80.0, 60.0)
    obs5 = create_rotated_rect(600.0, 1000.0, 80.0, 80.0, -15.0)
    obs6 = create_rotated_rect(200.0, 1000.0, 60.0, 100.0, 75.0)
    obs7 = create_rotated_rect(600.0, 600.0, 100.0, 100.0, 30.0)
    obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, obs7]
    
    # 3. Generate 6 terminal points
    terminals = [
        (100.0, 100.0),
        (1100.0, 100.0),
        (100.0, 1100.0),
        (1100.0, 1100.0),
        (600.0, 500.0),
        (600.0, 700.0)
    ]
    
    # 4. Construct the environment and routing grid
    grid_start_time = time.perf_counter()
    env = NonOrthogonalEnvironment(room, obstacles, terminals)
    grid_elapsed_ms = (time.perf_counter() - grid_start_time) * 1000.0
    
    print(f"Grid constructed successfully in {grid_elapsed_ms:.2f} ms.")
    print(f"Total grid nodes: {len(env.nodes)}")
    print("Running solvers...\n")
    
    results = []
    
    # 5. Sweep BendAwareKMB with different C_bend values
    c_bends = [0.0, 100.0, 500.0, 2000.0]
    for c_bend in c_bends:
        solver = BendAwareKMBSolver(env, C_bend=c_bend)
        t_start = time.perf_counter()
        res = solver.solve()
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        results.append((f"BendAwareKMB (C_bend={int(c_bend)})", res["weight"], res["turns"], t_elapsed))
        
    # 6. Run TurnCleanupSolver
    cleanup_solver = TurnCleanupSolver(env)
    t_start = time.perf_counter()
    res_cleanup = cleanup_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("TurnCleanupSolver", res_cleanup["weight"], res_cleanup["turns"], t_elapsed))
    
    # 7. Sweep BendAwareFastCorner with different C_bend values
    fc_c_bends = [100.0, 500.0]
    for c_bend in fc_c_bends:
        fc_solver = BendAwareFastCornerSolver(env, C_bend=c_bend)
        t_start = time.perf_counter()
        res_fc = fc_solver.solve()
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        results.append((f"BendAwareFastCorner (C_bend={int(c_bend)})", res_fc["weight"], res_fc["turns"], t_elapsed))
    # 8. Run BendAwareDualGraphKMBSolver with C_bend = 500
    dg_solver = BendAwareDualGraphKMBSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_dg = dg_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("BendAwareDualGraphKMB (C_bend=500)", res_dg["weight"], res_dg["turns"], t_elapsed))

    # 9. Run BendAwareStochasticKMBSolver with C_bend = 500
    stoch_solver = BendAwareStochasticKMBSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_stoch = stoch_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("BendAwareStochasticKMB (C_bend=500)", res_stoch["weight"], res_stoch["turns"], t_elapsed))

    # 11. Run BendAwareDualGraphGBFSSolver with C_bend = 500
    dg_gbfs_solver = BendAwareDualGraphGBFSSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_dg_gbfs = dg_gbfs_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("BendAwareDualGraphGBFS (C_bend=500)", res_dg_gbfs["weight"], res_dg_gbfs["turns"], t_elapsed))
    
    # 12. Run BendAwareDualGraphFastCornerSolver with C_bend = 500
    dg_fc_solver = BendAwareDualGraphFastCornerSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_dg_fc = dg_fc_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("BendAwareDualGraphFastCorner (C_bend=500)", res_dg_fc["weight"], res_dg_fc["turns"], t_elapsed))

    # 13. Run BendAwareDualGraphGBFSFastCornerSolver with C_bend = 500
    dg_gbfs_fc_solver = BendAwareDualGraphGBFSFastCornerSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_dg_gbfs_fc = dg_gbfs_fc_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("BendAwareDualGraphGBFSFastCorner (C_bend=500)", res_dg_gbfs_fc["weight"], res_dg_gbfs_fc["turns"], t_elapsed))

    # 14. Run StateExpandedSequentialFastCornerSolver with C_bend = 500
    se_seq_solver = StateExpandedSequentialFastCornerSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_se_seq = se_seq_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("StateExpandedSequentialFastCorner (C_bend=500)", res_se_seq["weight"], res_se_seq["turns"], t_elapsed))

    # 15. Run DualGraphSequentialFastCornerSolver with C_bend = 500
    dg_seq_solver = DualGraphSequentialFastCornerSolver(env, C_bend=500.0)
    t_start = time.perf_counter()
    res_dg_seq = dg_seq_solver.solve()
    t_elapsed = (time.perf_counter() - t_start) * 1000.0
    results.append(("DualGraphSequentialFastCorner (C_bend=500)", res_dg_seq["weight"], res_dg_seq["turns"], t_elapsed))
    
    # 7. Print the results table
    print("=" * 80)
    print(f"{'BEND-AWARE NON-ORTHOGONAL ROUTING COMPARISON':^80}")
    print("=" * 80)
    print(f"{'Solver / Config':<30} | {'Length (units)':<15} | {'Turn Count':<12} | {'Runtime (ms)':<12}")
    print("-" * 80)
    for name, length, turns, duration in results:
        print(f"{name:<30} | {length:<15.2f} | {turns:<12} | {duration:<12.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
