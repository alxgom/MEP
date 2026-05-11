# Multi-Net Routing Research Summary (v4.7)

## 1. Problem: Node Theft & Aggressive Rip-Up
We identified that the previous Hybrid Rip-Up implementation suffered from two main flaws:
1.  **Node Theft:** During the "Ideal Draft", a net could use another net's terminal as a Steiner point. When re-routing with `Hard_Lock`, that terminal became inaccessible to its owner, leading to avoidable failures.
2.  **Collateral Damage:** The rip-up was "Global"—it removed all segments touching any contested node/edge for ALL involved nets. This forced everyone to re-compete for the same resource, often leading back to the same collision.

## 2. Solution: Surgical Ownership & Pruning
We refactored the Hybrid solver to use a **Surgical Rip-Up** strategy:
- **Terminal Protection:** The environment now treats other nets' terminals as hard obstacles for the active net, preventing Node Theft at the source.
- **Ownership Arbitration:** For every contested resource (node or edge), the solver now elects a **Winner** based on Terminal Ownership and Bounding Box Area.
- **Surgical Pruning:** After ripping shared segments, the solver iteratively prunes dangling Steiner paths, ensuring visual and topological cleanliness.
- **Negotiated Hybrid (New):** Introduced a third variant that uses a **Soft Negotiated Draft** (congestion base = 2.0). Unlike the "Ideal" draft which produces purely selfish shortest paths, the Negotiated Draft is globally coordinated, resulting in a more distributed topology that avoids creating "bottleneck scars" even if the total length is slightly higher.

## 3. Results
- **Optimality Gap:** The Surgical (Ideal) solver achieves lengths within **1% of Global Permutation**.
- **Topology Diversity:** The Negotiated Hybrid provides a "Spreading" effect, making it highly suitable for 3D architectural piping where pipe density can cause physical installation issues.
- **Robustness:** Both Hybrid variants resolved all collision scenarios in our benchmark set.

## 4. Next Steps: Steiner Reconnection
Currently, the "Forest-Based" reconnection uses shortest-path MST to stitch islands. Integrating a full Steiner optimization during the "healing" phase would further reduce total network length.
