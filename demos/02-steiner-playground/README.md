# Demo #2: Steiner Point Playground

Interactive visualization of the Euclidean Steiner Tree Problem using a hybrid variational approach.

## Files

| File | Description |
|------|-------------|
| `steiner_solver.py` | Pure Python solver (no pygame). MST + gradient descent + SA + annihilation |
| `main.py` | Interactive Pygame visualizer — drag vertices, watch forces, see 120° angles |
| `headless_demo.py` | Headless matplotlib version — generates animated GIF |
| `steiner_interactive.py` | Generates `steiner_interactive.html` — browser-only version (vanilla JS) |
| `steiner_presets.json` | 15 preset geometries |
| `test_steiner.py` | 30 unit tests — all passing |
| `requirements.txt` | Dependencies: pygame, networkx, imageio, matplotlib |

## Controls

| Key | Action |
|-----|--------|
| N | Next algorithm step |
| A / Space | Auto-run optimization |
| R | Reset algorithm |
| C | Clear all terminals |
| F | Toggle fast mode |
| +/- | Add/remove Steiner point |
| G | Generate random terminals |
| , / . | Previous/next preset |
| L | Load selected preset |
| P | Print solver result to console |
| Scroll | Zoom in/out |
| Click+drag | Reposition vertex |

## Algorithm

1. **Initialize**: Place Steiner points randomly inside the bounding box of terminals
2. **MST Snap**: Build MST over all points (Kruskal's with DSU)
3. **Gradient Descent**: Each Steiner point moves along the sum of unit vectors toward its MST neighbors
4. **Simulated Annealing**: Thermal perturbation helps escape local minima
5. **Annihilation**: Nearby Steiner points merge (distance < 1% of bounding box diagonal)
6. **120° Verification**: At equilibrium, all Steiner points satisfy the Fermat condition (edges meet at 120°)

## Running

```bash
# Interactive Pygame demo
python main.py

# Headless (generates GIF, no display needed)
python headless_demo.py

# Generate browser HTML
python steiner_interactive.py
# Then serve: python -m http.server 8080
# Open http://localhost:8080/steiner_interactive.html

# Run tests
python -m pytest test_steiner.py -v
```

## Key Concepts Demonstrated

- **NP-hard Steiner Tree**: Unlike MST (polynomial), finding the optimal Steiner tree requires heuristic approaches
- **Physics analogy**: Steiner points as particles in zero-length springs; energy minimization ↔ cost minimization
- **Gradient descent**: Analytical gradient = sum of unit vectors toward MST neighbors
- **120° rule**: At the Fermat point, all three edges meet at exactly 120° angles
- **Simulated Annealing**: Metropolis criterion allows uphill moves to escape local minima
- **Annihilation**: Dynamic pruning of unnecessary Steiner points
- **MST lag**: Topology snaps discretely at each step (non-differentiable boundary)