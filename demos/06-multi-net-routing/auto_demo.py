"""
Headless Multi-Net Tournament Runner (v5.1)
===========================================
Database-backed benchmarking system that supports:
- Incremental simulation (runs only missing tests)
- Decoupled plotting from execution
- Modular healing comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from environment import MultiNetEnvironment
from solver import MultiNetSolver
from db_logger import RoutingDB

def run_tournament(scenario_name="Standard_3Net", force_rerun=False):
    db = RoutingDB()
    
    # 1. Setup Scenario
    np.random.seed(42) 
    nets = {
        "Net_1": np.random.randint(100, 700, (10, 2)),
        "Net_2": np.random.randint(100, 700, (10, 2)),
        "Net_3": np.random.randint(100, 700, (10, 2))
    }
    
    sid = db.save_scenario(scenario_name, nets)
    print(f"--- Tournament: {scenario_name} (ID: {sid}) ---")

    # 2. Define Solver Suite
    # (Solver Name, Method, Args)
    solvers_to_run = [
        ("Negotiated", "solve_negotiated", {}),
        ("Surgical_Simple", "solve_hybrid_ripup", {"healing_type": "simple"}),
        ("Surgical_Steiner", "solve_hybrid_ripup", {"healing_type": "steiner"}),
        ("NegHybrid_Simple", "solve_negotiated_hybrid", {"healing_type": "simple"}),
        ("NegHybrid_Steiner", "solve_negotiated_hybrid", {"healing_type": "steiner"}),
        ("GlobalPerm", "solve_best_permutation", {})
    ]

    env = MultiNetEnvironment(nets)
    solver = MultiNetSolver(env)

    # 3. Incremental Execution
    for name, method_name, kwargs in solvers_to_run:
        if db.check_result_exists(sid, name) and not force_rerun:
            print(f"  [SKIP] {name} (Result already in DB)")
            continue
        
        print(f"  [RUN]  {name}...")
        method = getattr(solver, method_name)
        res, weight, issues = method(**kwargs)
        db.save_result(sid, name, res, weight, issues)
        print(f"         Len: {weight:.1f} | Issues: {issues}")

    # 4. Decoupled Plotting
    plot_results(sid, db)
    db.close()

def plot_results(scenario_id: int, db: RoutingDB):
    name, nets = db.get_scenario(scenario_id)
    results = db.get_results_for_scenario(scenario_id)
    
    if not results:
        print("No results to plot.")
        return

    # Sort results to keep consistent order in plot
    results.sort(key=lambda x: x["solver_name"])
    
    n_cols = len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 6))
    if n_cols == 1: axes = [axes]
    
    plt.subplots_adjust(top=0.85)
    
    env = MultiNetEnvironment(nets) # For node coordinate lookup
    
    for idx, res in enumerate(results):
        ax = axes[idx]
        s_name = res["solver_name"]
        ax.set_title(f"{s_name}\nLen: {res['total_weight']:.1f} | Iss: {res['issues']}", fontweight='bold')
        ax.set_xlim(-100, 900)
        ax.set_ylim(-100, 900)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        sol_data = res["segments"]
        for net_idx, (net_name, data) in enumerate(sol_data.items()):
            net_colors = ["#2ecc71", "#3498db", "#9b59b6", "#f1c40f"]
            color = net_colors[net_idx % len(net_colors)]
            
            # Terminals
            t = nets[net_name]
            ax.scatter(t[:,0], t[:,1], c=color, s=30, zorder=5, edgecolors='white')
            
            # Segments
            for u, v in data["segments"]:
                p1, p2 = env.nodes[u], env.nodes[v]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2, alpha=0.8, zorder=2)
                
        ax.grid(True, linestyle=':', alpha=0.3)

    plt.suptitle(f"Tournament Comparison: {name} (v5.1)", fontsize=16, fontweight='bold')
    out_path = f"tournament_{name}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nDashboard saved to: {out_path}")

if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    run_tournament(force_rerun=force)
