"""
Live Gradio Statistical Dashboard (v2.5)
========================================
- Side-by-side topology visualizer with Steiner point highlighting.
- Statistical spread (shaded deviation).
- MST Gain comparison.
- Win Probability mean line.
- 30s refresh interval.
"""

import gradio as gr
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import numpy as np

def get_data():
    if not os.path.exists("benchmark_results.db"): return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    try:
        # Use timeout and Read-Only URI for concurrent access
        conn = sqlite3.connect("file:benchmark_results.db?mode=ro", uri=True, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        
        # 1. Per-Map Best Gains, MST Gaps and Win Probability
        df_maps = pd.read_sql_query("""
            SELECT c.n_terminals, c.config_id, 
                   ((c.greedy_length - min_stoch.min_l) / c.greedy_length * 100) as best_gain_over_greedy,
                   ((c.mst_length - c.greedy_length) / c.mst_length * 100) as greedy_gain_over_mst,
                   ((c.mst_length - min_stoch.min_l) / c.mst_length * 100) as stoch_gain_over_mst,
                   prob.win_prob
            FROM configurations c
            JOIN (SELECT config_id, MIN(stoch_length) as min_l FROM trials GROUP BY config_id) min_stoch 
            ON c.config_id = min_stoch.config_id
            JOIN (SELECT config_id, (CAST(SUM(is_winner) AS FLOAT) / COUNT(trial_id)) * 100 as win_prob FROM trials GROUP BY config_id) prob
            ON c.config_id = prob.config_id
        """, conn)

        # 2. Per-N Stats (Mean and Std Dev)
        df_stats = df_maps.groupby("n_terminals").agg({
            "best_gain_over_greedy": ["mean", "std"],
            "greedy_gain_over_mst": "mean",
            "stoch_gain_over_mst": "mean",
            "win_prob": "mean"
        }).reset_index()
        df_stats.columns = ["n_terminals", "mean_gain", "std_gain", "greedy_mst_mean", "stoch_mst_mean", "win_prob_mean"]
        df_stats = df_stats.fillna(0)

        conn.close()
        return df_maps, df_stats, df_maps # probability data now in df_maps
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def get_map_data(config_id):
    if not config_id: return None, None, None, None, 0, 0
    conn = sqlite3.connect("benchmark_results.db")
    res = conn.execute("SELECT terminals_json, obstacles_json, greedy_segments_json, greedy_length FROM configurations WHERE config_id=?", (config_id,)).fetchone()
    best_stoch_res = conn.execute("""
        SELECT stoch_length, stoch_segments_json FROM trials 
        WHERE config_id=? ORDER BY stoch_length ASC LIMIT 1
    """, (config_id,)).fetchone()
    conn.close()
    if res:
        best_l, best_seg = best_stoch_res if best_stoch_res else (0, None)
        return json.loads(res[0]), json.loads(res[1]), json.loads(res[2]), \
               (json.loads(best_seg) if best_seg else None), res[3], best_l
    return None, None, None, None, 0, 0

def update_global_plots():
    df_maps, df_stats, df_prob = get_data()
    if df_maps.empty: return go.Figure(), go.Figure(), go.Figure(), "Waiting for data...", gr.update()

    # PLOT 1: Savings over Greedy (with Shaded Deviation)
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Scatter(
        x=pd.concat([df_stats["n_terminals"], df_stats["n_terminals"][::-1]]),
        y=pd.concat([df_stats["mean_gain"] + df_stats["std_gain"], (df_stats["mean_gain"] - df_stats["std_gain"])[::-1]]),
        fill='toself', fillcolor='rgba(59, 130, 246, 0.1)', line_color='rgba(255,255,255,0)',
        name="Deviation (Spread)", showlegend=True
    ))
    fig_imp.add_trace(go.Scatter(x=df_maps["n_terminals"], y=df_maps["best_gain_over_greedy"], mode="markers", 
                                 marker=dict(color="rgba(30, 64, 175, 0.3)", size=6), name="Individual Maps"))
    fig_imp.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["mean_gain"], mode="lines+markers", 
                                 line=dict(color="#1d4ed8", width=3), name="Mean Gain"))
    fig_imp.update_layout(title="Stochastic Savings over Greedy (%)", xaxis_title="N", yaxis_title="Savings %", template="plotly_white")

    # PLOT 2: Steiner Gain over MST Baseline
    fig_mst = go.Figure()
    fig_mst.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["greedy_mst_mean"], name="Greedy Steiner", 
                                 line=dict(color="red", width=2, dash='dot')))
    fig_mst.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["stoch_mst_mean"], name="Stochastic Steiner", 
                                 line=dict(color="#2ecc71", width=3)))
    fig_mst.update_layout(title="Efficiency Gain vs. Baseline MST (%)", xaxis_title="N", yaxis_title="Reduction in Length (%)", template="plotly_white")

    # PLOT 3: Win Probability (with Mean Line)
    fig_prob = go.Figure()
    # Scatter points
    fig_prob.add_trace(go.Scatter(x=df_prob["n_terminals"], y=df_prob["win_prob"], mode="markers", 
                                  marker=dict(color="rgba(16, 185, 129, 0.3)", size=10), name="Maps"))
    # Mean line
    fig_prob.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["win_prob_mean"], mode="lines+markers",
                                  line=dict(color="#059669", width=3), name="Mean Prob"))
    fig_prob.update_layout(title="Stochastic Win Probability", xaxis_title="N", yaxis_title="Win Prob %", template="plotly_white")
    
    config_choices = [f"ID {cid} (N={n})" for cid, n in zip(df_prob['config_id'], df_prob['n_terminals'])]
    status_text = f"Logged Maps: {len(df_prob)} | Current block: N={df_prob['n_terminals'].max() if not df_prob.empty else '?'}"
    return fig_imp, fig_mst, fig_prob, status_text, gr.update(choices=config_choices)

def create_base(title, length, obstacles, terminals):
    fig = go.Figure()
    for obs in obstacles:
        fig.add_shape(type="rect", x0=obs['min_x'], y0=obs['min_y'], x1=obs['max_x'], y1=obs['max_y'], 
                      fillcolor="red", opacity=0.15, line_width=0)
    t = np.array(terminals)
    fig.add_trace(go.Scatter(x=t[:,0], y=t[:,1], mode='markers', marker=dict(color='#3b82f6', size=7), name="Terminals"))
    fig.update_layout(title=f"{title}<br>Length: {length:.2f}", xaxis=dict(range=[0,800]), yaxis=dict(range=[0,800]), 
                      width=480, height=450, template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def render_side_by_side(selection):
    if not selection: return go.Figure(), go.Figure(), "Select a map"
    config_id = int(selection.split(' ')[1])
    terminals, obstacles, greedy_seg, best_seg, greedy_l, best_l = get_map_data(config_id)
    
    def extract_steiner_junctions(segments, terminals):
        if not segments: return np.array([]).reshape(0, 2)
        from collections import Counter
        point_counts = Counter()
        for u, v in segments:
            point_counts[tuple(np.round(u, 3))] += 1
            point_counts[tuple(np.round(v, 3))] += 1
        steiner = []
        t_set = {tuple(np.round(t, 3)) for t in terminals}
        for pt, degree in point_counts.items():
            if pt not in t_set and degree >= 3:
                steiner.append(pt)
        return np.array(steiner)

    # 1. Greedy Plot
    f_greedy = create_base("Deterministic Greedy", greedy_l, obstacles, terminals)
    if greedy_seg:
        for s in greedy_seg:
            pts = np.array(s)
            f_greedy.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='lines', line=dict(color='red', width=2, dash='dot')))
        s_pts = extract_steiner_junctions(greedy_seg, terminals)
        if len(s_pts) > 0:
            f_greedy.add_trace(go.Scatter(x=s_pts[:,0], y=s_pts[:,1], mode='markers', 
                                         marker=dict(color='#e67e22', size=10, symbol='square'), name="Junctions"))

    # 2. Stochastic Plot (with Overlay)
    f_stoch = create_base("Best Stochastic Winner", best_l, obstacles, terminals)
    if greedy_seg: # Ghost Greedy
        for s in greedy_seg:
            pts = np.array(s)
            f_stoch.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='lines', line=dict(color='rgba(255,0,0,0.1)', width=1.5, dash='dot')))
    if best_seg:
        for s in best_seg:
            pts = np.array(s)
            f_stoch.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='lines', line=dict(color='#2ecc71', width=3)))
        s_pts = extract_steiner_junctions(best_seg, terminals)
        if len(s_pts) > 0:
            f_stoch.add_trace(go.Scatter(x=s_pts[:,0], y=s_pts[:,1], mode='markers', 
                                         marker=dict(color='#e67e22', size=10, symbol='square'), name="Junctions"))
    
    saving = greedy_l - best_l
    report = f"Stochastic Gain: {saving:.2f} units ({(saving/greedy_l*100):.2f}%)"
    return f_greedy, f_stoch, report

with gr.Blocks(title="Manhattan Steiner Rigor v2.5") as demo:
    gr.Markdown("# 🚀 Manhattan Steiner Rigor Dashboard v2.5")
    with gr.Row():
        status = gr.Textbox(label="Benchmark Status", interactive=False)
        timer = gr.Timer(30)
    with gr.Row():
        plot_savings = gr.Plot(label="Gain over Greedy")
        plot_mst_gain = gr.Plot(label="Gain over MST")
        plot_prob = gr.Plot(label="Win Probability")
    gr.Markdown("---")
    gr.Markdown("## 🗺️ Side-by-Side Analysis (with Steiner Highlighting)")
    with gr.Row():
        map_dd = gr.Dropdown(label="Select configuration ID", choices=[])
        map_info = gr.Textbox(label="Delta Metrics", interactive=False)
    with gr.Row():
        gv = gr.Plot(); sv = gr.Plot()
    timer.tick(update_global_plots, outputs=[plot_savings, plot_mst_gain, plot_prob, status, map_dd])
    map_dd.change(render_side_by_side, inputs=map_dd, outputs=[gv, sv, map_info])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
