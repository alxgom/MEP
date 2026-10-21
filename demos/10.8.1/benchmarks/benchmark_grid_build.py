"""Compare cold/warm grid builds with numeric and GEOS intersections."""

import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import statistics
import sys
from time import perf_counter

import numpy as np


def benchmark(cases=2, repeats=3):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with redirect_stdout(io.StringIO()):
        import main as app
    from mep_routing.graphs import GraphMode, hybrid
    from mep_routing.graphs.cells import cached_cell_template

    numeric = hybrid._orthogonal_intersections
    modes = (GraphMode.HANNAN, GraphMode.CELL_SKELETON, GraphMode.HANNAN_CELL)
    rows = []
    sources = [(0, i) for i in range(min(cases, len(app.REAL_DWELLING_SCENARIOS)))] + [(1, 0)]
    try:
        for source, case in sources:
            random.seed(42)
            np.random.seed(42)
            app.dwelling_source_idx, app.real_scenario_idx = source, case
            app.graph_type_idx, app.router_backend_idx = 1, 0
            with redirect_stdout(io.StringIO()):
                app.dwelling_runtime.generate()
            lifecycle = app.workbench_state.dwelling.workspace.lifecycle
            lifecycle.hannan_template(shift_walls=True)
            pose = app.workbench_state.installation("sal").pose
            x, y, angle = pose.as_tuple()
            for dx, dy, rotation in ((0, 0, 0), (100, 0, 0), (0, 0, 90)):
                pose.set((x + dx, y + dy), angle + rotation)
                pins = app.routing_controller.machine_pins(*pose.as_tuple())
                reference = {}
                for repeat in range(repeats):
                    paths = ("geos", "numeric") if repeat % 2 == 0 else ("numeric", "geos")
                    for path in paths:
                        hybrid._orthogonal_intersections = numeric if path == "numeric" else hybrid._geometry_intersections
                        for mode in modes:
                            cached_cell_template.cache_clear()
                            hybrid._cached_join.cache_clear()
                            times = []
                            for _ in range(2):
                                started = perf_counter()
                                result = lifecycle.build_selected(mode, pins, pose.angle_degrees)
                                times.append((perf_counter() - started) * 1000)
                                runtime = result.runtime
                                if mode not in reference:
                                    reference[mode] = (runtime.nodes.copy(), runtime.edge_list.copy(), runtime.env.cells)
                                expected_nodes, expected_edges, expected_cells = reference[mode]
                                np.testing.assert_array_equal(runtime.nodes, expected_nodes)
                                assert runtime.edge_list == expected_edges and runtime.env.cells == expected_cells
                            rows.append({"source": "real" if source == 0 else "synthetic",
                                         "case": app.current_scenario_label, "pose": pose.as_tuple(),
                                         "repeat": repeat, "path": path, "grid": mode.name,
                                         "cold_ms": round(times[0], 3), "warm_ms": round(times[1], 3),
                                         "nodes": len(runtime.nodes), "edges": len(runtime.edge_list)})
    finally:
        hybrid._orthogonal_intersections = numeric
        cached_cell_template.cache_clear()
        hybrid._cached_join.cache_clear()
    summary = {}
    for mode in modes:
        summary[mode.name] = {}
        for path in ("geos", "numeric"):
            selected = [r for r in rows if r["grid"] == mode.name and r["path"] == path]
            summary[mode.name][path] = {"samples": len(selected)}
            for phase in ("cold", "warm"):
                times = sorted(r[f"{phase}_ms"] for r in selected)
                summary[mode.name][path].update({
                    f"{phase}_median_ms": round(statistics.median(times), 3),
                    f"{phase}_p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
                    f"{phase}_max_ms": times[-1],
                })
    return {"note": "Grid lifecycle builds including runtime assembly; excludes routing, rendering, dwelling preparation and static Hanan axes. Cold clears cell-template and union caches; warm immediately repeats the same pose. Alternating implementation order. Every output matches exact nodes, edge records and cell overlays. Machine-dependent timings.",
            "summary": summary, "samples": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error("cases and repeats must be positive")
    result = benchmark(args.cases, args.repeats)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
