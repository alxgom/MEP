"""
Dashboard Query & Generator
===========================
Queries the SQLite database and generates an interactive Plotly dashboard.
"""

import sqlite3
import json
import pandas as pd
import os

def generate_dashboard(db_path="benchmark_results.db", output_file="scaling_dashboard.html"):
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    
    # 1. Query for Improvement Scaling (Line Chart)
    # We want the BEST improvement found for each config, averaged across the 5 maps of that N.
    query_imp = """
    SELECT 
        c.n_terminals,
        AVG((c.greedy_length - min_stoch.min_l) / c.greedy_length * 100) as avg_improvement
    FROM configurations c
    JOIN (
        SELECT config_id, MIN(stoch_length) as min_l 
        FROM trials 
        GROUP BY config_id
    ) min_stoch ON c.config_id = min_stoch.config_id
    GROUP BY c.n_terminals
    ORDER BY c.n_terminals
    """
    df_imp = pd.read_sql_query(query_imp, conn)

    # 2. Query for Success Probability (Scatter Plot)
    # Probability of beating greedy for every single config
    query_prob = """
    SELECT 
        c.n_terminals,
        c.config_id,
        (CAST(SUM(t.is_winner) AS FLOAT) / COUNT(t.trial_id)) * 100 as win_prob
    FROM configurations c
    JOIN trials t ON c.config_id = t.config_id
    GROUP BY c.config_id
    ORDER BY c.n_terminals
    """
    df_prob = pd.read_sql_query(query_prob, conn)
    conn.close()

    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Manhattan Steiner Scaling Analysis</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-50 p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold text-gray-800 mb-2">Manhattan Steiner Scaling Dashboard</h1>
        <p class="text-gray-600 mb-8 border-b pb-4">Analyzing Stochastic 'Kick' heuristic vs. Deterministic Greedy ($N=40$ to $100$)</p>

        <div class="grid grid-cols-1 gap-8">
            <!-- Improvement Chart -->
            <div class="bg-white p-6 rounded-xl shadow-sm border">
                <h2 class="text-xl font-semibold mb-4 text-blue-600">Mean Efficiency Gain vs. Complexity</h2>
                <p class="text-sm text-gray-500 mb-4">How much shorter is the 'Best of 50' result compared to a single Greedy run?</p>
                <div id="impChart" style="height: 500px;"></div>
            </div>

            <!-- Probability Scatter -->
            <div class="bg-white p-6 rounded-xl shadow-sm border">
                <h2 class="text-xl font-semibold mb-4 text-green-600">Trial Success Probability</h2>
                <p class="text-sm text-gray-500 mb-4">What is the % chance that a random trial will beat the greedy result? (Each point is a unique map)</p>
                <div id="probChart" style="height: 500px;"></div>
            </div>
        </div>
        
        <footer class="mt-8 text-center text-gray-400 text-xs italic">
            Generated from benchmark_results.db
        </footer>
    </div>

    <script>
        const impData = IMP_DATA_PLACEHOLDER;
        const probData = PROB_DATA_PLACEHOLDER;

        // 1. Improvement Line Chart
        Plotly.newPlot('impChart', [{
            x: impData.map(d => d.n_terminals),
            y: impData.map(d => d.avg_improvement),
            mode: 'lines+markers',
            name: 'Avg Gain',
            line: { shape: 'spline', color: '#3b82f6', width: 4 },
            marker: { size: 10, color: '#1d4ed8' },
            fill: 'tozeroy',
            fillcolor: 'rgba(59, 130, 246, 0.1)'
        }], {
            xaxis: { title: 'Number of Terminals (N)', gridcolor: '#f1f5f9' },
            yaxis: { title: 'Length Savings over Greedy (%)', gridcolor: '#f1f5f9' },
            margin: { t: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        });

        // 2. Probability Scatter Plot
        Plotly.newPlot('probChart', [{
            x: probData.map(d => d.n_terminals),
            y: probData.map(d => d.win_prob),
            mode: 'markers',
            marker: {
                size: 12,
                color: '#22c55e',
                opacity: 0.6,
                line: { color: 'white', width: 1 }
            },
            name: 'Maps'
        }], {
            xaxis: { title: 'Number of Terminals (N)', gridcolor: '#f1f5f9' },
            yaxis: { title: 'Win Probability (%)', gridcolor: '#f1f5f9', range: [0, 105] },
            margin: { t: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        });
    </script>
</body>
</html>
    """
    
    final_html = html_template.replace("IMP_DATA_PLACEHOLDER", json.dumps(df_imp.to_dict(orient="records")))
    final_html = final_html.replace("PROB_DATA_PLACEHOLDER", json.dumps(df_prob.to_dict(orient="records")))
    
    with open(output_file, "w") as f:
        f.write(final_html)
    
    print(f"Dashboard generated: {output_file}")

if __name__ == "__main__":
    generate_dashboard()
