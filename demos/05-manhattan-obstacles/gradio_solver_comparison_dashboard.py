"""
Interactive Solver Comparison Dashboard (v1.0)
==============================================
Allows comparative analysis of all 8 solvers:
1. MST (baseline)
2. KMB
3. Greedy
4. FastCorner
5. Prune
6. StochKMB
7. AnisoKMB
8. MonteCarlo

Visualizes actual routed length, solver times, and Steiner counts.
"""

import os
import sys
import time
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
from typing import List, Tuple, Dict, Any

# Resolve imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from environment import Obstacle, GridEnvironment
from solver import ObstacleSteinerSolver
from run_solver_benchmark_v2 import make_scaled_map, BASE_W, BASE_H, N_REF, OBSTACLE_FRACS

DB_PATH = os.path.join(parent_dir, "solver_benchmark_v2.db")

SOLVER_MAP = {
    "MST":            lambda s: s.solve_mst(),
    "KMB":            lambda s: s.solve_kou(),
    "Greedy":         lambda s: s.solve_greedy(),
    "FastCorner":     lambda s: s.solve_fast_corner(),
    "Prune":          lambda s: s.solve_prune(),
    "StochKMB":       lambda s: s.solve_stochastic_kou(n_trials=20, perturbation=0.10),
    "Aniso_Isotropic":lambda s: s.solve_anisotropic_kou(w_x=1.0, w_y=1.0, n_trials=1, sigma=0.0),
    "Aniso_Asymmetric":lambda s: s.solve_anisotropic_kou(w_x=1.0, w_y=2.0, n_trials=1, sigma=0.0),
    "Aniso_Stochastic":lambda s: s.solve_anisotropic_kou(w_x=1.5, w_y=1.5, n_trials=15, sigma=0.3),
    "MonteCarlo":     lambda s: s.solve_monte_carlo(),
}

# -----------------------------------------------------------------------------
# Database Queries
# -----------------------------------------------------------------------------
def get_benchmark_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT n, solver, true_length, mst_length, saving_pct, time_ms, n_steiners FROM results", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading DB: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Plotly Chart Builders
# -----------------------------------------------------------------------------
def build_comparison_charts(selected_solvers: List[str]) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure, pd.DataFrame, str]:
    df = get_benchmark_data()
    if df.empty or not selected_solvers:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return empty_fig, empty_fig, empty_fig, empty_fig, pd.DataFrame(), "⚠️ Database is empty or no solvers selected."

    # Filter by selected solvers
    df_filtered = df[df['solver'].isin(selected_solvers)]
    if df_filtered.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return empty_fig, empty_fig, empty_fig, empty_fig, pd.DataFrame(), "⚠️ No data matching selection."

    # Compute averages
    df_grouped = df_filtered.groupby(['n', 'solver']).mean().reset_index()

    # Color palette map for solvers
    colors = {
        "MST": "#95a5a6",
        "KMB": "#3498db",
        "Greedy": "#e74c3c",
        "FastCorner": "#2ecc71",
        "Prune": "#e67e22",
        "StochKMB": "#9b59b6",
        "Aniso_Isotropic": "#1abc9c",
        "Aniso_Asymmetric": "#16a085",
        "Aniso_Stochastic": "#27ae60",
        "MonteCarlo": "#f1c40f"
    }

    # Chart 1: True Length vs N
    fig_len = go.Figure()
    for solver in selected_solvers:
        df_s = df_grouped[df_grouped['solver'] == solver]
        if not df_s.empty:
            fig_len.add_trace(go.Scatter(
                x=df_s['n'], y=df_s['true_length'],
                mode='lines+markers', name=solver,
                line=dict(color=colors.get(solver, "#ffffff"), width=3),
                marker=dict(size=8)
            ))
    fig_len.update_layout(
        title="<b>Average Routed Length</b><br>Lower is better (Coordinate units)",
        xaxis=dict(title="Terminal Count (N)", gridcolor='#2c2c35', showgrid=True),
        yaxis=dict(title="Routed Length", gridcolor='#2c2c35', showgrid=True),
        template="plotly_dark", plot_bgcolor='#111115', paper_bgcolor='#191921',
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(bgcolor='rgba(25, 25, 33, 0.8)', bordercolor='#2c2c35', borderwidth=1)
    )

    # Chart 2: Savings % vs N
    fig_sav = go.Figure()
    for solver in selected_solvers:
        df_s = df_grouped[df_grouped['solver'] == solver]
        if not df_s.empty:
            fig_sav.add_trace(go.Scatter(
                x=df_s['n'], y=df_s['saving_pct'],
                mode='lines+markers', name=solver,
                line=dict(color=colors.get(solver, "#ffffff"), width=3),
                marker=dict(size=8)
            ))
    fig_sav.update_layout(
        title="<b>Average Length Savings vs. MST (%)</b><br>Higher is better (Percentage)",
        xaxis=dict(title="Terminal Count (N)", gridcolor='#2c2c35', showgrid=True),
        yaxis=dict(title="Savings (%)", gridcolor='#2c2c35', showgrid=True),
        template="plotly_dark", plot_bgcolor='#111115', paper_bgcolor='#191921',
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(bgcolor='rgba(25, 25, 33, 0.8)', bordercolor='#2c2c35', borderwidth=1)
    )

    # Chart 3: Solving Time (ms) vs N
    fig_time = go.Figure()
    for solver in selected_solvers:
        df_s = df_grouped[df_grouped['solver'] == solver]
        if not df_s.empty:
            fig_time.add_trace(go.Scatter(
                x=df_s['n'], y=df_s['time_ms'],
                mode='lines+markers', name=solver,
                line=dict(color=colors.get(solver, "#ffffff"), width=3),
                marker=dict(size=8)
            ))
    fig_time.update_layout(
        title="<b>Solving Time (ms)</b><br>Log scale recommended for Greedy vs others",
        xaxis=dict(title="Terminal Count (N)", gridcolor='#2c2c35', showgrid=True),
        yaxis=dict(title="Time (ms)", type="log", gridcolor='#2c2c35', showgrid=True),
        template="plotly_dark", plot_bgcolor='#111115', paper_bgcolor='#191921',
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(bgcolor='rgba(25, 25, 33, 0.8)', bordercolor='#2c2c35', borderwidth=1)
    )

    # Chart 4: Steiner Count vs N
    fig_st = go.Figure()
    for solver in selected_solvers:
        df_s = df_grouped[df_grouped['solver'] == solver]
        if not df_s.empty:
            fig_st.add_trace(go.Scatter(
                x=df_s['n'], y=df_s['n_steiners'],
                mode='lines+markers', name=solver,
                line=dict(color=colors.get(solver, "#ffffff"), width=3),
                marker=dict(size=8)
            ))
    fig_st.update_layout(
        title="<b>Steiner Node Count (Junctions)</b><br>Normalized (excludes collinear degree-2 pass-throughs)",
        xaxis=dict(title="Terminal Count (N)", gridcolor='#2c2c35', showgrid=True),
        yaxis=dict(title="Junction Count", gridcolor='#2c2c35', showgrid=True),
        template="plotly_dark", plot_bgcolor='#111115', paper_bgcolor='#191921',
        margin=dict(l=50, r=30, t=70, b=50),
        legend=dict(bgcolor='rgba(25, 25, 33, 0.8)', bordercolor='#2c2c35', borderwidth=1)
    )

    # Format summary table
    summary_df = df_grouped[['n', 'solver', 'true_length', 'saving_pct', 'time_ms', 'n_steiners']].copy()
    summary_df.columns = ["N", "Solver", "Avg Length", "Savings %", "Time (ms)", "Junctions"]
    summary_df = summary_df.round({"Avg Length": 1, "Savings %": 2, "Time (ms)": 1, "Junctions": 1})

    status_msg = f"✅ Database loaded successfully. Showing averages across {df['n'].count() // len(selected_solvers)} records."
    return fig_len, fig_sav, fig_time, fig_st, summary_df, status_msg

# -----------------------------------------------------------------------------
# Spatial Map Drawing helper
# -----------------------------------------------------------------------------
def make_plotly_map(env: GridEnvironment, solution: Dict[str, Any], title: str, elapsed_ms: float, color: str) -> go.Figure:
    fig = go.Figure()

    # 1. Background grid edges
    row, col = env.adj_matrix.nonzero()
    edge_x, edge_y = [], []
    for u, v in zip(row, col):
        if u < v:
            p1, p2 = env.nodes[u], env.nodes[v]
            edge_x.extend([p1[0], p2[0], None])
            edge_y.extend([p1[1], p2[1], None])
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode='lines',
        line=dict(color='rgba(80, 80, 96, 0.1)', width=1),
        hoverinfo='skip', showlegend=False
    ))

    # 2. Obstacles
    for o in env.obstacles:
        fig.add_shape(
            type="rect", x0=o.min_x, y0=o.min_y, x1=o.max_x, y1=o.max_y,
            fillcolor="#34495e", opacity=0.5,
            line=dict(color="#2c3e50", width=1.5), layer="below"
        )

    # 3. Steiner Tree Edges
    tree_x, tree_y = [], []
    for u, v in solution["segments"]:
        p1, p2 = env.nodes[u], env.nodes[v]
        tree_x.extend([p1[0], p2[0], None])
        tree_y.extend([p1[1], p2[1], None])
    fig.add_trace(go.Scatter(
        x=tree_x, y=tree_y, mode='lines',
        line=dict(color=color, width=4),
        name='Routed Pipeline'
    ))

    # 4. Sinks (Terminals)
    fig.add_trace(go.Scatter(
        x=env.terminals[:, 0], y=env.terminals[:, 1], mode='markers',
        marker=dict(size=8, color="#3498db", line=dict(color='white', width=1.2)),
        name='Terminals'
    ))

    # 5. Steiner Points (junctions)
    steiner_nodes = env.nodes[solution["steiner_indices"]]
    if len(steiner_nodes) > 0:
        fig.add_trace(go.Scatter(
            x=steiner_nodes[:, 0], y=steiner_nodes[:, 1], mode='markers',
            marker=dict(size=10, color='#e67e22', symbol='star', line=dict(color='white', width=0.8)),
            name='Junctions (Steiner)'
        ))

    # Layout
    W = float(env.nodes[:, 0].max() - env.nodes[:, 0].min())
    H = float(env.nodes[:, 1].max() - env.nodes[:, 1].min())
    
    fig.update_layout(
        title=f"<b>{title}</b><br>Length: {solution['weight']:.1f} | Time: {elapsed_ms:.1f}ms | Junctions: {len(solution['steiner_indices'])}",
        plot_bgcolor='#111115', paper_bgcolor='#191921',
        font=dict(color='#f1f1f5', family='Fira Code, monospace'),
        xaxis=dict(showgrid=True, gridcolor='#2c2c35', zeroline=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=True, gridcolor='#2c2c35', zeroline=False),
        width=530, height=500,
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(bgcolor='rgba(25,25,33,0.8)', bordercolor='#2c2c35', x=0.01, y=0.99)
    )
    return fig

# -----------------------------------------------------------------------------
# Dynamic Solver Comparison on Map
# -----------------------------------------------------------------------------
def solve_and_compare(n: int, seed: int, solver_a: str, solver_b: str) -> Tuple[go.Figure, go.Figure, str]:
    # 1. Generate the map (Preserving shape, constant density, scaled obstacles)
    terminals, obstacles, W, H = make_scaled_map(n, seed)
    env = GridEnvironment(terminals, obstacles)

    # 2. Run Solver A
    sa = ObstacleSteinerSolver(env)
    t0 = time.time()
    res_a = SOLVER_MAP[solver_a](sa)
    time_a = (time.time() - t0) * 1000

    # 3. Run Solver B
    sb = ObstacleSteinerSolver(env)
    t0 = time.time()
    res_b = SOLVER_MAP[solver_b](sb)
    time_b = (time.time() - t0) * 1000

    # Colors
    color_a = "#ff4d4d" # Red
    color_b = "#2ecc71" # Green

    # Create maps
    fig_a = make_plotly_map(env, res_a, f"Solver A: {solver_a}", time_a, color_a)
    fig_b = make_plotly_map(env, res_b, f"Solver B: {solver_b}", time_b, color_b)

    # Compare
    len_diff = res_a['weight'] - res_b['weight']
    faster_diff = time_a - time_b

    # Length outcome
    if np.isclose(len_diff, 0):
        len_report = f"⚖️ **Path lengths are identical ({res_a['weight']:.1f}).**"
    elif len_diff > 0:
        len_report = f"🏆 **{solver_b} is shorter** by **{len_diff:.1f}** units ({(len_diff / res_a['weight'] * 100):.2f}% savings)."
    else:
        len_report = f"🏆 **{solver_a} is shorter** by **{-len_diff:.1f}** units ({(-len_diff / res_b['weight'] * 100):.2f}% savings)."

    # Speed outcome
    if np.isclose(faster_diff, 0, atol=0.5):
        speed_report = "⚖️ **Solving speeds are practically identical.**"
    elif faster_diff > 0:
        speed_report = f"⚡ **{solver_b} is faster** by **{faster_diff:.1f} ms** ({time_a / time_b:.1f}x speedup)."
    else:
        speed_report = f"⚡ **{solver_a} is faster** by **{-faster_diff:.1f} ms** ({time_b / time_a:.1f}x speedup)."

    report_md = f"""
### ⚖️ Side-by-Side Comparison Summary
* **Room Scale:** {W:.0f}x{H:.0f} | **Density:** {n / (W*H) * 10000:.2f} terminals per 10k px²

1. **Routing Length:**
   * {solver_a} Length: `{res_a['weight']:.1f}`
   * {solver_b} Length: `{res_b['weight']:.1f}`
   * **Result:** {len_report}

2. **Computation Time:**
   * {solver_a} Time: `{time_a:.1f} ms`
   * {solver_b} Time: `{time_b:.1f} ms`
   * **Result:** {speed_report}

3. **Junctions (Fittings):**
   * {solver_a} Junctions: `{len(res_a['steiner_indices'])}`
   * {solver_b} Junctions: `{len(res_b['steiner_indices'])}`
"""
    return fig_a, fig_b, report_md

# -----------------------------------------------------------------------------
# Gradio Design layout
# -----------------------------------------------------------------------------
css = """
.gradio-container { background-color: #0f172a !important; }
.gr-dataframe, table {
    border-collapse: collapse !important;
    width: 100% !important;
    color: #ffffff !important;
}
th {
    color: #ffffff !important;
    background-color: #1e3a8a !important;
    font-weight: 800 !important;
    font-size: 1.1em !important;
    border: 1px solid #475569 !important;
    padding: 10px !important;
}
td, .cell, .cell-wrap {
    color: #ffffff !important;
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    padding: 8px !important;
    font-size: 1.0em !important;
}
tr:nth-child(even) td, tr:nth-child(even) .cell {
    background-color: #334155 !important;
}
td *, .cell * {
    color: #ffffff !important;
}
"""

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate",
).set(
    body_text_color="#f1f1f5",
    body_text_color_subdued="#a0a0b0",
    background_fill_primary="#121216",
    background_fill_secondary="#191921",
    block_background_fill="#191921",
    block_label_text_color="#ffffff",
    input_background_fill="#22222c",
    input_border_color="#2c2c35",
    input_border_color_focus="#3b82f6"
)

with gr.Blocks(theme=theme, css=css) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px; border-bottom: 2px solid #3b82f6; padding-bottom: 15px;">
        <h1 style="color: #3b82f6; font-family: 'Fira Code', monospace; font-size: 2.2em; font-weight: bold; margin-bottom: 5px;">
            🎛️ MEP Multi-Solver Steiner Comparison Dashboard
        </h1>
        <p style="color: #a0a0b0; font-size: 1.1em; font-family: 'Segoe UI', sans-serif;">
            Compare actual routed lengths, latencies, and Steiner junction counts across all 8 heuristic/stochastic solvers.
        </p>
    </div>
    """)

    with gr.Tabs():
        # --- TAB 1: HISTORICAL BENCHMARK STATS ---
        with gr.Tab("📊 Benchmark Statistical Sweeps"):
            gr.Markdown("""
            ### 📈 Multi-Solver Performance Analysis
            Select which solvers you wish to overlay on the charts. The data is pulled directly from the **constant-density v2 database** (`solver_benchmark_v2.db`).
            """)

            with gr.Row():
                db_status = gr.Textbox(label="Database Status", value="Checking...", interactive=False, scale=3)
                refresh_btn = gr.Button("🔄 Refresh Database Averages", variant="secondary", scale=1)

            with gr.Row():
                solver_checkboxes = gr.CheckboxGroup(
                    label="Select Solvers to Plot",
                    choices=list(SOLVER_MAP.keys()),
                    value=list(SOLVER_MAP.keys()),
                    interactive=True
                )

            with gr.Row():
                plot_len = gr.Plot(label="Avg Routed Length")
                plot_sav = gr.Plot(label="Avg Savings vs MST (%)")
            with gr.Row():
                plot_time = gr.Plot(label="Avg Solving Time (ms)")
                plot_st = gr.Plot(label="Avg Steiner Nodes")

            gr.Markdown("---")
            gr.Markdown("### 📋 Average Results Data Table")
            results_table = gr.DataFrame(
                headers=["N", "Solver", "Avg Length", "Savings %", "Time (ms)", "Junctions"],
                datatype=["number", "str", "number", "number", "number", "number"],
                interactive=False
            )

        # --- TAB 2: INTERACTIVE SIDE-BY-SIDE ROUTING ---
        with gr.Tab("🎮 Interactive Map Visualizer"):
            gr.Markdown("""
            ### 🗺️ Side-by-Side Map Comparisons
            Select the terminal count ($N$) and seed, choose **any two solvers**, and click **Regenerate & Solve** to run both algorithms side-by-side on a newly scaled room!
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### ⚙️ Parameters")
                    n_slider = gr.Slider(minimum=10, maximum=50, step=10, value=20, label="Terminal Count (N)")
                    seed_input = gr.Number(value=42, label="Random Map Seed", precision=0)
                    
                    solver_a_dd = gr.Dropdown(choices=list(SOLVER_MAP.keys()), value="Greedy", label="Select Solver A (Red)")
                    solver_b_dd = gr.Dropdown(choices=list(SOLVER_MAP.keys()), value="FastCorner", label="Select Solver B (Green)")
                    
                    solve_btn = gr.Button("⚡ Regenerate & Solve", variant="primary")
                    gr.Markdown("---")
                    compare_results_md = gr.Markdown("*Click 'Regenerate & Solve' to compare solvers.*")

                with gr.Column(scale=3):
                    with gr.Row():
                        plot_map_a = gr.Plot(label="Solver A Output")
                        plot_map_b = gr.Plot(label="Solver B Output")

    # Wire up events
    def update_all_plots(selected_solvers):
        fig_l, fig_sav, fig_t, fig_st, df_tbl, status = build_comparison_charts(selected_solvers)
        return fig_l, fig_sav, fig_t, fig_st, df_tbl, status

    refresh_btn.click(
        fn=update_all_plots,
        inputs=[solver_checkboxes],
        outputs=[plot_len, plot_sav, plot_time, plot_st, results_table, db_status]
    )

    solver_checkboxes.change(
        fn=update_all_plots,
        inputs=[solver_checkboxes],
        outputs=[plot_len, plot_sav, plot_time, plot_st, results_table, db_status]
    )

    solve_btn.click(
        fn=solve_and_compare,
        inputs=[n_slider, seed_input, solver_a_dd, solver_b_dd],
        outputs=[plot_map_a, plot_map_b, compare_results_md]
    )

    # Initial load
    demo.load(
        fn=update_all_plots,
        inputs=[solver_checkboxes],
        outputs=[plot_len, plot_sav, plot_time, plot_st, results_table, db_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7869)
