"""Compare grid choices at identical poses, without opening a Pygame window."""

from contextlib import redirect_stdout
import argparse
import io
import json
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter

import numpy as np


def benchmark(cases=2, repeats=2, *, include_cells=False, include_hybrid=False, include_hanan_cells=False):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with redirect_stdout(io.StringIO()):
        import main as app
    from mep_routing.graphs import GraphMode
    modes = (GraphMode.REGULAR, GraphMode.HANNAN, GraphMode.MEDIAL_AXIS)
    if include_cells:
        modes += (GraphMode.CELL_SKELETON,)
    if include_hybrid:
        modes += (GraphMode.HANNAN_MEDIAL,)
    if include_hanan_cells:
        modes += (GraphMode.HANNAN_CELL,)
    rows = []
    sources = [(0, index) for index in range(min(cases, len(app.REAL_DWELLING_SCENARIOS)))] + [(1, 0)]
    for source, case in sources:
        random.seed(42)
        np.random.seed(42)
        app.dwelling_source_idx, app.real_scenario_idx = source, case
        app.graph_type_idx, app.router_backend_idx = 1, 0
        with redirect_stdout(io.StringIO()):
            app.dwelling_runtime.generate()
        pose = app.workbench_state.installation("sal").pose
        x, y, angle = pose.as_tuple()
        for dx, dy, rotation in ((0, 0, 0), (100, 0, 0), (0, 0, 90)):
            pose.set((x + dx, y + dy), angle + rotation)
            for mode in modes:
                app.graph_type_idx = int(mode)
                with redirect_stdout(io.StringIO()):
                    app.routing_controller.build_grid(machine_pins=app.routing_controller.machine_pins(*pose.as_tuple()))
                for backend in (0, 1):
                    app.router_backend_idx = backend
                    for repeat in range(repeats + 1):
                        with redirect_stdout(io.StringIO()):
                            started = perf_counter()
                            routes, status, _, _ = app.routing_controller.solve_installations()
                            elapsed = (perf_counter() - started) * 1000
                        if repeat == 0:
                            continue
                        workspace = app.workbench_state.dwelling.workspace
                        rows.append({
                            "source": "real" if source == 0 else "synthetic", "case": app.current_scenario_label,
                            "pose": pose.as_tuple(), "grid": mode.name, "backend": backend,
                            "scene_ms": round(elapsed, 3), "nodes": len(workspace.nodes),
                            "edges": len(workspace.edge_list), "routes": len(routes or ()),
                            "turns": app.application.count_solution_turns(routes or ()), "status": status,
                        })
    summary = {}
    for mode in modes:
        selected = [row for row in rows if row["grid"] == mode.name]
        times = sorted(row["scene_ms"] for row in selected)
        summary[mode.name] = {
            "samples": len(selected), "successful": sum(row["status"].startswith("Success") for row in selected),
            "median_ms": round(statistics.median(times), 3),
            "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
            "median_nodes": statistics.median(row["nodes"] for row in selected),
        }
    return {"boundary_sample_spacing_mm": app.demo_config.medial_sample_spacing_mm,
            "cell_boundary_resolution_mm": app.demo_config.cell_boundary_resolution_mm,
            "cell_min_area_mm2": app.demo_config.cell_min_area_mm2,
            "wall_grid_offset_mm": app.demo_config.routing_wall_clearance_mm,
            "cell_wall_grid": "hanan_shifted_axes",
            "strategy": app.demo_config.routing_strategies[app.routing_strategy_idx],
            "note": "Warm full Sal scene solves; fixed poses; timings are machine dependent. Failures are retained.",
            "summary": summary, "samples": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-cells", action="store_true", help="Also compare the Figure-5-style orthogonal cell skeleton")
    parser.add_argument("--include-hybrid", action="store_true", help="Also compare Hanan plus Voronoi with local links")
    parser.add_argument("--include-hanan-cells", action="store_true", help="Also compare the merged Hanan and cell grid")
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error("cases and repeats must be positive")
    result = benchmark(args.cases, args.repeats, include_cells=args.include_cells, include_hybrid=args.include_hybrid,
                       include_hanan_cells=args.include_hanan_cells)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
