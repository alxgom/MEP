# MEP Ventilation Router Dashboard (Demo 10.5)

An interactive, real-time Mechanical, Electrical, and Plumbing (MEP) ventilation router and auto-placement visualizer built with Pygame, Shapely, and SciPy.

This visualizer resolves multiple turn-penalized ventilation paths (Shaft, Kitchen, Bathrooms, Toilet, Washroom) from the machine's ports to their respective room terminals and extraction shafts, while providing real-time KPI graphs.

## 🚀 Execution Instructions

### Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install pygame numpy shapely scipy
```

### Run the Visualizer
Navigate to the directory and run the visualizer using the ComfyUI virtual environment or your local Python interpreter:
```bash
cd c:\DEV\MEP\demos\10.5
& "C:\DEV\Image generation\ComfyUI\.venv\Scripts\python.exe" main.py
```

---

## 🎮 Interactive Controls

### Machine Manipulation
* **Drag Machine:** Hover over the orange/red extraction machine, click and drag with the **Left Mouse Button**.
* **Mouse Scroll:** Scroll the mouse wheel (`Scroll Up` to rotate CCW, `Scroll Down` to rotate CW) while dragging or hovering near the machine.
* **`R` Key:** Rotate the machine 90° CW.

### Dashboard Options
* **`A` Key:** Cycle through automatic machine placement modes.
* **`V` Key:** Toggle the auto-placement scoring heatmap overlay.
* **`H` Key:** Toggle the heatmap scale representation (**Linear 75% Saturation** vs. **Logarithmic Scale**).
* **`W` Key:** Toggle auto-placement weight profile (**Default weights** prioritizing short exhaust runs vs. **Equal weights** prioritizing pure topological proximity).
* **`C` Key:** Cycle the path solver strategy:
  1. *Greedy (Dual-Sort)*
  2. *First Fit (Permutation Backtracking)*
  3. *Best Fit (Exhaustive Permutation Search)*
  4. *Negotiated Congestion (A*-based iterative congestion resolver)*
* **`Tab` Key:** Cycle the underlying routing grid:
  1. *Regular Grid* (200mm resolution)
  2. *Hannan Grid* (built from room/obstacle boundaries)
  3. *Hannan + Shifted Nodes* (adds shifted hallway pathways)
* **`G` Key:** Toggle the visibility of the underlying grid mesh nodes and edges.
* **`Space` Key:** Generate a completely new random apartment dwelling layout.

---

## 📊 Analytics & Plots
The right panel contains four real-time plots showing:
1. **Total Duct Length (m)**
2. **Total Solution Cost Score**
3. **Total Turns**
4. **Turns / Routed Length (turns/m)**

* **Comparison to Baseline:** All metrics show the percentage change compared to the initial auto-placed state.
* **Minimum Value Tracking:** The historical minimum values reached during exploration are highlighted on each plot.
* **Event Markers:** Key events (Strategy switches, grid toggles, rotations, scale toggles, weights toggles, automatic placements) are plotted as color-coded vertical line markers.

---

## 🛠️ Key Architectural Enhancements
* **Multi-Side Shaft Snaps:** Routes to/from the closest boundary node of the shaft polygon within 500mm and connects the duct straight to the shaft center core centroid.
* **Obstacle Collision Avoidance:** Rejects machine automatic placements overlapping with columns/shafts.
* **Negotiated Congestion Weighting:** Multiplies large duct (Kitchen/Shaft) congestion weights by 0.35 in Strategy 4, allowing smaller ducts to route around them.
