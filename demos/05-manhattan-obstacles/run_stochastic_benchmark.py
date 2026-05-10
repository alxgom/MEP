"""
Stochastic Kick Spread Analysis
===============================
Runs 50 independent trials of the Stochastic Fast Corner Kick on a 40-terminal obstacle map.
Analyzes the variance and potential gains over the deterministic greedy baseline.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver

def run_trial(args):
    terminals_list, obstacles_data, seed, is_stochastic = args
    terminals = np.array(terminals_list)
    obstacles = [Obstacle(o[0], o[1], o[2], o[3]) for o in obstacles_data]
    
    # 1. Build environment (needs to be done in each process)
    env = GridEnvironment(terminals, obstacles)
    solver = ObstacleSteinerSolver(env)
    
    # 2. Run solver
    np.random.seed(seed) # Ensure trial diversity
    res = solver.solve_fast_corner(stochastic=is_stochastic, temperature=0.1)
    
    return res["weight"]

def main():
    print("Initializing Stochastic Spread Analysis (N=40, 50 trials)...")
    
    # 1. Setup Scenario
    np.random.seed(42)
    terminals = np.random.rand(40, 2) * 800
    obstacles_data = [
        [200, 200, 300, 600],
        [500, 100, 600, 400],
        [100, 400, 400, 500]
    ]
    
    # 2. Run Deterministic Baseline
    print("Running Deterministic Baseline...")
    baseline_args = (terminals.tolist(), obstacles_data, 42, False)
    baseline_weight = run_trial(baseline_args)
    print(f"Baseline Weight: {baseline_weight:.2f}")

    # 3. Run Stochastic Batch in Parallel
    num_trials = 50
    print(f"Running {num_trials} Stochastic Trials in parallel...")
    
    trial_args = [(terminals.tolist(), obstacles_data, i, True) for i in range(num_trials)]
    
    with ProcessPoolExecutor() as executor:
        weights = list(executor.map(run_trial, trial_args))

    # 4. Statistical Analysis
    weights = np.array(weights)
    min_w = np.min(weights)
    max_w = np.max(weights)
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    
    improvement = (baseline_weight - min_w) / baseline_weight * 100
    
    print("\n--- Results ---")
    print(f"Min Weight:    {min_w:.2f} ({improvement:.2f}% improvement over baseline)")
    print(f"Mean Weight:   {mean_w:.2f}")
    print(f"Max Weight:    {max_w:.2f}")
    print(f"Std Deviation: {std_w:.4f}")

    # 5. Visualize Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(baseline_weight, color='red', linestyle='dashed', linewidth=2, label=f'Baseline ({baseline_weight:.2f})')
    plt.axvline(min_w, color='green', linestyle='dashed', linewidth=2, label=f'Best Found ({min_w:.2f})')
    
    plt.title(f"Spread Analysis: Stochastic Fast Corner (N=40, {num_trials} trials)")
    plt.xlabel("Total Tree Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    output_plot = "stochastic_spread.png"
    plt.savefig(output_plot, dpi=150)
    print(f"\nSpread plot saved to {output_plot}")

if __name__ == "__main__":
    main()
