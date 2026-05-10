"""
Main Tournament Runner
======================
Executes the full benchmarking suite and generates a report.
"""

import os
import json
import pandas as pd
from framework import Tournament
from visualizer import plot_comparison

def generate_markdown_report(results_file="tournament_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Summary by Algorithm
    summary = df.groupby("algorithm").agg({
        "length": "mean",
        "gap_pct": "mean",
        "time": "mean",
        "steiner_count": "mean",
        "max_120_dev": "mean"
    }).sort_values("gap_pct")
    
    report = "# Steiner Tournament Benchmarking Report\n\n"
    report += "## Executive Summary\n"
    report += summary.to_markdown() + "\n\n"
    
    report += "## Detailed Results per Case\n"
    for suite in df["suite"].unique():
        report += f"### Suite: {suite}\n"
        suite_df = df[df["suite"] == suite]
        pivot = suite_df.pivot(index="case", columns="algorithm", values="length")
        report += pivot.to_markdown() + "\n\n"
        
    with open("REPORT.md", "w") as f:
        f.write(report)
    print("Report saved to REPORT.md")

def main():
    print("Starting Steiner Tournament...")
    t = Tournament()
    t.run()
    
    print("\nGenerating Report...")
    try:
        generate_markdown_report()
    except Exception as e:
        print(f"Could not generate MD report (is pandas installed?): {e}")
        # Fallback to simple report if pandas fails
    
    print("\nGenerating Visualizations for key cases...")
    # Plot first case of each suite
    with open("tournament_results.json", "r") as f:
        data = json.load(f)
        suites_plotted = set()
        for r in data:
            if r["suite"] not in suites_plotted:
                plot_comparison(r["case"])
                suites_plotted.add(r["suite"])

    print("\nTournament Complete!")

if __name__ == "__main__":
    main()
