import os
import sys
import time
import numpy as np
import plotly.graph_objects as go
import gradio as gr
from shapely.geometry import Polygon

# Resolve imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from environment import NonOrthogonalEnvironment
from solver import (
    BendAwareKMBSolver, TurnCleanupSolver, BendAwareFastCornerSolver,
    BendAwareDualGraphKMBSolver, BendAwareStochasticKMBSolver, BendAwarePruneSolver,
    BendAwareDualGraphGBFSSolver, BendAwareDualGraphFastCornerSolver,
    BendAwareDualGraphGBFSFastCornerSolver, StateExpandedSequentialFastCornerSolver,
    DualGraphSequentialFastCornerSolver
)
from main import create_rotated_rect

# -----------------------------------------------------------------------------
# Initialize the fixed test environment
# -----------------------------------------------------------------------------
room = Polygon([
    (0, 0), (1200, 0), (1200, 400), (800, 400), (800, 800), (1200, 800),
    (1200, 1200), (0, 1200), (0, 800), (400, 800), (400, 400), (0, 400)
])

# Placed 7 rotated obstacles
obs1 = create_rotated_rect(250.0, 200.0, 100.0, 60.0, 15.0)
obs2 = create_rotated_rect(600.0, 200.0, 80.0, 120.0, -30.0)
obs3 = create_rotated_rect(1000.0, 200.0, 60.0, 60.0, 45.0)
obs4 = create_rotated_rect(1000.0, 1000.0, 120.0, 80.0, 60.0)
obs5 = create_rotated_rect(600.0, 1000.0, 80.0, 80.0, -15.0)
obs6 = create_rotated_rect(200.0, 1000.0, 60.0, 100.0, 75.0)
obs7 = create_rotated_rect(600.0, 600.0, 100.0, 100.0, 30.0)
obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, obs7]

terminals = [
    (100.0, 100.0),
    (1100.0, 100.0),
    (100.0, 1100.0),
    (1100.0, 1100.0),
    (600.0, 500.0),
    (600.0, 700.0)
]

env = NonOrthogonalEnvironment(room, obstacles, terminals)

# -----------------------------------------------------------------------------
# Plotly Map Builder
# -----------------------------------------------------------------------------
def make_plotly_map(solution_segs, title: str, elapsed_ms: float, color: str) -> go.Figure:
    fig = go.Figure()

    # 1. Draw Room boundary
    rx, ry = zip(*room.exterior.coords)
    fig.add_trace(go.Scatter(
        x=list(rx), y=list(ry),
        mode='lines',
        fill='toself',
        fillcolor='rgba(44, 44, 84, 0.1)',
        line=dict(color='#8892b0', width=2),
        name="L-Shaped Room",
        hoverinfo='skip'
    ))

    # 2. Draw Rotated Obstacles
    for i, obs in enumerate(obstacles):
        ox, oy = zip(*obs.exterior.coords)
        fig.add_trace(go.Scatter(
            x=list(ox), y=list(oy),
            mode='lines',
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.25)',
            line=dict(color='#e74c3c', width=1.5),
            name=f"Obstacle {i+1}",
            hoverinfo='name'
        ))

    # 3. Draw Grid Nodes (gray waypoints)
    fig.add_trace(go.Scatter(
        x=env.nodes[:, 0], y=env.nodes[:, 1],
        mode='markers',
        marker=dict(size=4, color='rgba(255, 255, 255, 0.15)'),
        name="Routing Grid Nodes",
        hoverinfo='skip'
    ))

    # 4. Draw Steiner Tree segments
    edge_x = []
    edge_y = []
    for p1, p2 in solution_segs:
        edge_x.extend([p1[0], p2[0], None])
        edge_y.extend([p1[1], p2[1], None])
        
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color=color, width=4),
        name="Steiner Path",
        hoverinfo='none'
    ))

    # 5. Draw Terminals
    tx = [t[0] for t in terminals]
    ty = [t[1] for t in terminals]
    fig.add_trace(go.Scatter(
        x=tx, y=ty,
        mode='markers',
        marker=dict(
            size=12,
            color='#f1c40f',
            line=dict(color='#ffffff', width=2)
        ),
        name="Terminals",
        hoverinfo='text',
        text=[f"Terminal: ({x}, {y})" for x, y in zip(tx, ty)]
    ))

    # Configure layout
    fig.update_layout(
        title=f"<b>{title}</b><br>Solved in {elapsed_ms:.2f} ms",
        xaxis=dict(showgrid=True, gridcolor='#2c2c35', range=[-50, 1250]),
        yaxis=dict(showgrid=True, gridcolor='#2c2c35', range=[-50, 1250], scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        plot_bgcolor='#111115',
        paper_bgcolor='#191921',
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# -----------------------------------------------------------------------------
# Gradio Control Logic
# -----------------------------------------------------------------------------
def solve_and_render(solver_type: str, C_bend_val: float):
    t_start = time.perf_counter()
    if solver_type == "BendAwareKMB":
        solver = BendAwareKMBSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareKMB (C_bend={int(C_bend_val)})"
        color = "#3498db" # Blue
    elif solver_type == "BendAwareFastCorner":
        solver = BendAwareFastCornerSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareFastCorner (C_bend={int(C_bend_val)})"
        color = "#9b59b6" # Purple
    elif solver_type == "BendAwareDualGraphKMB":
        solver = BendAwareDualGraphKMBSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareDualGraphKMB (C_bend={int(C_bend_val)})"
        color = "#e67e22" # Orange
    elif solver_type == "BendAwareStochasticKMB":
        solver = BendAwareStochasticKMBSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareStochasticKMB (C_bend={int(C_bend_val)})"
        color = "#f1c40f" # Yellow
    elif solver_type == "BendAwarePrune":
        solver = BendAwarePruneSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwarePrune (C_bend={int(C_bend_val)})"
        color = "#e74c3c" # Red
    elif solver_type == "BendAwareDualGraphGBFS":
        solver = BendAwareDualGraphGBFSSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareDualGraphGBFS (C_bend={int(C_bend_val)})"
        color = "#16a085" # Teal
    elif solver_type == "BendAwareDualGraphFastCorner":
        solver = BendAwareDualGraphFastCornerSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareDualGraphFastCorner (C_bend={int(C_bend_val)})"
        color = "#d35400" # Dark Orange
    elif solver_type == "BendAwareDualGraphGBFSFastCorner":
        solver = BendAwareDualGraphGBFSFastCornerSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"BendAwareDualGraphGBFSFastCorner (C_bend={int(C_bend_val)})"
        color = "#c0392b" # Dark Red
    elif solver_type == "StateExpandedSequentialFastCorner":
        solver = StateExpandedSequentialFastCornerSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"StateExpandedSequentialFastCorner (C_bend={int(C_bend_val)})"
        color = "#2c3e50" # Dark Slate
    elif solver_type == "DualGraphSequentialFastCorner":
        solver = DualGraphSequentialFastCornerSolver(env, C_bend=C_bend_val)
        res = solver.solve()
        title_str = f"DualGraphSequentialFastCorner (C_bend={int(C_bend_val)})"
        color = "#8e44ad" # Dark Purple
    else:
        solver = TurnCleanupSolver(env)
        res = solver.solve()
        title_str = "TurnCleanupSolver"
        color = "#2ecc71" # Green
        
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    
    fig = make_plotly_map(res["segments"], title_str, elapsed_ms, color)
    
    stats_md = f"""
    ### 📊 Routing Metrics
    *   **Total Pipe Length:** `{res["weight"]:.2f} units`
    *   **Total $90^\circ$ Bends (Turns):** `{res["turns"]}`
    *   **Computation Time:** `{elapsed_ms:.2f} ms`
    """
    
    return fig, stats_md

# -----------------------------------------------------------------------------
# Layout Definition
# -----------------------------------------------------------------------------
theme = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="#0d0e12",
    block_background_fill="#15171e",
    block_border_color="#272a35",
    button_primary_background_fill="#1e51a4",
    button_primary_background_fill_hover="#2b6ad4"
)

css = """
footer {visibility: hidden !important;}
"""

with gr.Blocks(theme=theme, css=css) as demo:
    gr.Markdown("""
    # 📐 Demo 08: Bend-Aware Non-Orthogonal Routing Visualizer
    Compare turn-penalized Steiner routing in a room containing rotated obstacles.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Solver Configuration")
            solver_dd = gr.Dropdown(
                choices=[
                    "BendAwareKMB", "BendAwareFastCorner", "TurnCleanupSolver", 
                    "BendAwareDualGraphKMB", "BendAwareStochasticKMB", "BendAwarePrune", 
                    "BendAwareDualGraphGBFS", "BendAwareDualGraphFastCorner", 
                    "BendAwareDualGraphGBFSFastCorner", "StateExpandedSequentialFastCorner", 
                    "DualGraphSequentialFastCorner"
                ],
                value="BendAwareKMB",
                label="Select Solver"
            )
            
            c_bend_slider = gr.Slider(
                minimum=0.0,
                maximum=2000.0,
                step=100.0,
                value=500.0,
                label="Bend Penalty (C_bend)",
                info="Applies to all solvers except TurnCleanupSolver"
            )
            
            run_btn = gr.Button("⚡ Route Network", variant="primary")
            stats_panel = gr.Markdown("### 📊 Routing Metrics\n*Click Route Network to see stats.*")
            
        with gr.Column(scale=3):
            map_plot = gr.Plot(label="Routing Visualizer Map")

    # Interactive behavior for slider visibility
    def toggle_slider(solver):
        return gr.update(visible=(solver != "TurnCleanupSolver"))
        
    solver_dd.change(
        fn=toggle_slider,
        inputs=[solver_dd],
        outputs=[c_bend_slider]
    )

    run_btn.click(
        fn=solve_and_render,
        inputs=[solver_dd, c_bend_slider],
        outputs=[map_plot, stats_panel]
    )
    
    # Trigger initial load
    demo.load(
        fn=solve_and_render,
        inputs=[solver_dd, c_bend_slider],
        outputs=[map_plot, stats_panel]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7870)
