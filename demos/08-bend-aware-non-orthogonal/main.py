import time
import numpy as np
import shapely.affinity
from shapely.geometry import Polygon
from environment import NonOrthogonalEnvironment
from solver import BendAwareKMBSolver, TurnCleanupSolver

def create_rotated_rect(cx: float, cy: float, w: float, h: float, angle_deg: float) -> Polygon:
    """Creates a rectangle centered at (cx, cy) rotated by angle_deg degrees."""
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy + h/2
    rect = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return shapely.affinity.rotate(rect, angle_deg, origin='center')

def main():
    print("Initializing environment...")
    
    # 1. Initialize L-shaped room
    room = Polygon([
        (0, 0),
        (1000, 0),
        (1000, 500),
        (500, 500),
        (500, 1000),
        (0, 1000)
    ])
    
    # 2. Place 3 rotated obstacles inside the room
    # Obstacle 1: bottom-left area
    obs1 = create_rotated_rect(250.0, 250.0, 100.0, 150.0, 30.0)
    # Obstacle 2: bottom-right area
    obs2 = create_rotated_rect(750.0, 250.0, 120.0, 80.0, -45.0)
    # Obstacle 3: top-left area
    obs3 = create_rotated_rect(250.0, 750.0, 80.0, 120.0, 15.0)
    obstacles = [obs1, obs2, obs3]
    
    # 3. Generate terminal points
    terminals = [
        (100.0, 100.0),  # Bottom-Left
        (900.0, 100.0),  # Bottom-Right
        (100.0, 900.0),  # Top-Left
        (400.0, 400.0)   # Middle junction region
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
