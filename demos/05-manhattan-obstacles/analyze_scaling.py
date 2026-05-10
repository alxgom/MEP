"""
Statistical Scaling Analysis
===========================
Queries benchmark_results.db to analyze how stochastic success scales with N.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    conn = sqlite3.connect("benchmark_results.db")
    
    # 1. Join Tables to get full dataset
    query = """
    SELECT 
        c.n_terminals,
        c.config_id,
        c.greedy_length,
        t.stoch_length,
        t.is_winner
    FROM configurations c
    JOIN trials t ON c.config_id = t.config_id
    """
    df = pd.read_sql_query(query, conn)
    
    # 2. Calculate Statistics per N
    # Group by N_terminals
    scaling = df.groupby("n_terminals").apply(lambda x: pd.Series({
        "win_rate": (x["is_winner"].sum() / len(x)) * 100,
        "avg_greedy": x["greedy_length"].mean(),
        "avg_stoch": x["stoch_length"].mean(),
        # Best possible gain per N (across all trials for all configs of that N)
        "max_improvement": ((x.groupby("config_id")["greedy_length"].mean() - 
                             x.groupby("config_id")["stoch_length"].min()).mean())
    })).reset_index()
    
    print("\n--- Scaling Analysis Results ---")
    print(scaling)

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.patch.set_facecolor("#f8f9fa")

    # Plot 1: Win Probability Scaling
    ax1.plot(scaling["n_terminals"], scaling["win_rate"], marker='o', linewidth=3, color='#3498db')
    ax1.fill_between(scaling["n_terminals"], scaling["win_rate"], alpha=0.1, color='#3498db')
    ax1.set_title("Stochastic Win Probability vs. Terminal Count (N)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Number of Terminals (N)")
    ax1.set_ylabel("Win Probability (%)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # Plot 2: Average Length Improvement
    ax2.bar(scaling["n_terminals"], scaling["max_improvement"], width=3, color='#2ecc71', alpha=0.7)
    ax2.set_title("Average Length Savings over Greedy vs. N", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Number of Terminals (N)")
    ax2.set_ylabel("Mean Savings (Units)")
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("scaling_analysis.png", dpi=150)
    print("\nScaling analysis plot saved to scaling_analysis.png")
    
    conn.close()

if __name__ == "__main__":
    main()
