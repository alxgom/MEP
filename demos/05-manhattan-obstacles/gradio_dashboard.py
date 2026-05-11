"""
Live Gradio Statistical Dashboard (v2.6 SAFE MODE)
=================================================
- Defensive coding to prevent blank pages.
- Port 7865.
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
    if not os.path.exists("benchmark_results.db"): 
        return None, None, None
    try:
        # Use simple sqlite3 without complex URI for safety
        conn = sqlite3.connect("benchmark_results.db", timeout=20)
        conn.execute("PRAGMA journal_mode=WAL")
        
        # 1. Main map data
        df_maps = pd.read_sql_query("""
            SELECT c.n_terminals, c.config_id, c.mst_length, c.greedy_length,
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

        # 2. Stats per N
        df_stats = df_maps.groupby("n_terminals").agg({
            "best_gain_over_greedy": ["mean", "std"],
            "greedy_gain_over_mst": "mean",
            "stoch_gain_over_mst": "mean",
            "win_prob": "mean"
        }).reset_index()
        df_stats.columns = ["n_terminals", "mean_gain", "std_gain", "greedy_mst_mean", "stoch_mst_mean", "win_prob_mean"]
        df_stats = df_stats.fillna(0)

        conn.close()
        return df_maps, df_stats, df_maps
    except Exception as e:
        print(f"DEBUG Error: {e}")
        return None, None, None

def update_global_plots():
    df_maps, df_stats, df_prob = get_data()
    
    if df_maps is None or df_maps.empty:
        return go.Figure(), go.Figure(), go.Figure(), "No data in database yet...", gr.update(choices=[])

    try:
        # Plot 1: Savings
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Scatter(x=df_maps["n_terminals"], y=df_maps["best_gain_over_greedy"], mode="markers", 
                                     marker=dict(color="rgba(30, 64, 175, 0.3)", size=6), name="Maps"))
        fig_imp.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["mean_gain"], mode="lines+markers", 
                                     line=dict(color="#1d4ed8", width=3), name="Mean"))
        fig_imp.update_layout(title="Savings over Greedy (%)", template="plotly_white")

        # Plot 2: MST Gain
        fig_mst = go.Figure()
        fig_mst.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["greedy_mst_mean"], name="Greedy", line=dict(dash='dot')))
        fig_mst.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["stoch_mst_mean"], name="Stoch", line=dict(width=3)))
        fig_mst.update_layout(title="Gain vs MST (%)", template="plotly_white")

        # Plot 3: Win Prob
        fig_prob = go.Figure()
        fig_prob.add_trace(go.Scatter(x=df_prob["n_terminals"], y=df_prob["win_prob"], mode="markers", marker=dict(opacity=0.4)))
        fig_prob.add_trace(go.Scatter(x=df_stats["n_terminals"], y=df_stats["win_prob_mean"], mode="lines"))
        fig_prob.update_layout(title="Win Probability (%)", template="plotly_white")
        
        choices = [f"ID {cid} (N={n})" for cid, n in zip(df_prob['config_id'], df_prob['n_terminals'])]
        return fig_imp, fig_mst, fig_prob, f"Maps loaded: {len(df_prob)}", gr.update(choices=choices)
    except Exception as e:
        print(f"Plotting Error: {e}")
        return go.Figure(), go.Figure(), go.Figure(), f"Error: {e}", gr.update(choices=[])

def get_map_data(config_id):
    try:
        conn = sqlite3.connect("benchmark_results.db")
        res = conn.execute("SELECT terminals_json, obstacles_json, greedy_segments_json, greedy_length FROM configurations WHERE config_id=?", (config_id,)).fetchone()
        best_stoch_res = conn.execute("SELECT stoch_length, stoch_segments_json FROM trials WHERE config_id=? ORDER BY stoch_length ASC LIMIT 1", (config_id,)).fetchone()
        conn.close()
        if res:
            return json.loads(res[0]), json.loads(res[1]), json.loads(res[2]), json.loads(best_stoch_res[1]), res[3], best_stoch_res[0]
    except: pass
    return None, None, None, None, 0, 0

def render_side_by_side(selection):
    if not selection: return go.Figure(), go.Figure(), "Select a map"
    cid = int(selection.split(' ')[1])
    terminals, obstacles, greedy_seg, best_seg, greedy_l, best_l = get_map_data(cid)
    
    def create_base(title, length):
        fig = go.Figure()
        for obs in obstacles:
            fig.add_shape(type="rect", x0=obs['min_x'], y0=obs['min_y'], x1=obs['max_x'], y1=obs['max_y'], fillcolor="red", opacity=0.1)
        t = np.array(terminals)
        fig.add_trace(go.Scatter(x=t[:,0], y=t[:,1], mode='markers', marker=dict(size=6)))
        fig.update_layout(title=f"{title}<br>Len: {length:.1f}", width=400, height=400, template="plotly_white", showlegend=False)
        return fig

    f1 = create_base("Greedy", greedy_l)
    if greedy_seg:
        for s in greedy_seg:
            pts = np.array(s)
            f1.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='lines', line=dict(color='red', width=1)))

    f2 = create_base("Stochastic", best_l)
    if best_seg:
        for s in best_seg:
            pts = np.array(s)
            f2.add_trace(go.Scatter(x=pts[:,0], y=pts[:,1], mode='lines', line=dict(color='green', width=2)))

    return f1, f2, f"Gain: {greedy_l - best_l:.2f}"

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Steiner Stats Debug View (v2.6)")
    gr.Markdown("Attempting to load data from `benchmark_results.db`... If this is blank, the database is locked or empty.")
    
    with gr.Row():
        status = gr.Textbox(label="DB Status")
        refresh_btn = gr.Button("Manual Refresh")

    with gr.Row():
        p1 = gr.Plot(); p2 = gr.Plot(); p3 = gr.Plot()
        
    gr.Markdown("---")
    map_dd = gr.Dropdown(label="Map Selection")
    with gr.Row():
        gv = gr.Plot(); sv = gr.Plot()
        
    # Wire up events
    refresh_btn.click(update_global_plots, outputs=[p1, p2, p3, status, map_dd])
    map_dd.change(render_side_by_side, inputs=map_dd, outputs=[gv, sv, status])
    demo.load(update_global_plots, outputs=[p1, p2, p3, status, map_dd])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865)
