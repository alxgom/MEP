"""Repeatable full-scene Sal baseline/hard-length comparison without a window.

Run with the demo's Pygame interpreter. External dwelling-export inputs are
the same as main.py; timings include graph refresh, weights and all Sal routes.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter


def benchmark(cases=1, repeats=3):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    with redirect_stdout(io.StringIO()):
        import main as demo
    from mep_routing.routing import piece_search
    from mep_routing.routing.search import SearchBackend

    search = piece_search.run_piece_length_astar
    search_stats = []

    def measured_search(*args, **kwargs):
        stats = piece_search.PieceSearchStats()
        kwargs["stats"] = stats
        result = search(*args, **kwargs)
        search_stats.append(stats)
        return result

    piece_search.run_piece_length_astar = measured_search
    poses = ((0, 0, 0), (100, 0, 0), (-100, 0, 0), (0, 100, 0), (0, -100, 0), (0, 0, 90))
    backends = (SearchBackend.STATE_ASTAR, SearchBackend.LINE_GRAPH_ASTAR,
                SearchBackend.STATE_ASTAR_MIN_PIECES, SearchBackend.LINE_GRAPH_ASTAR_MIN_PIECES)
    samples = []
    try:
        for case in range(min(cases, len(demo.REAL_DWELLING_SCENARIOS) or 1)):
            demo.real_scenario_idx = case
            demo.router_backend_idx = 0
            with redirect_stdout(io.StringIO()):
                demo.dwelling_runtime.generate()
            pose = demo.workbench_state.installation("sal").pose
            x, y, angle = pose.as_tuple()
            for dx, dy, rotation in poses:
                pose.set((x + dx, y + dy), angle + rotation)
                for backend in backends:
                    demo.router_backend_idx = int(backend)
                    # Warm each graph/pose/backend combination before timing.
                    with redirect_stdout(io.StringIO()):
                        demo.routing_controller.solve_installations()
                    for _ in range(repeats):
                        search_stats.clear()
                        with redirect_stdout(io.StringIO()):
                            started = perf_counter()
                            routes, status, elapsed_ms, nodes = demo.routing_controller.solve_installations()
                            scene_ms = (perf_counter() - started) * 1000
                        samples.append({
                            "case": demo.current_scenario_label,
                            "pose": pose.as_tuple(), "backend": backend.name,
                            "scene_ms": scene_ms, "solver_ms": elapsed_ms,
                            "route_count": len(routes or ()), "route_nodes": nodes,
                            "short_pieces": demo.routing_controller.count_short_pieces(routes) if routes else None,
                            "status": status,
                            "expanded": sum(item.expanded for item in search_stats),
                            "generated": sum(item.generated for item in search_stats),
                            "max_labels_per_state": max((item.max_labels_per_state for item in search_stats), default=0),
                        })
    finally:
        piece_search.run_piece_length_astar = search
    summaries = {}
    for backend in backends:
        rows = [item for item in samples if item["backend"] == backend.name]
        times = sorted(item["scene_ms"] for item in rows)
        if not times:
            continue
        summaries[backend.name] = {
            "samples": len(rows), "successful": sum(bool(item["route_count"]) for item in rows),
            "median_ms": statistics.median(times),
            "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
            "max_ms": max(times), "over_200_ms": sum(value > 200 for value in times),
            "max_labels_per_state": max(item["max_labels_per_state"] for item in rows),
            "max_short_pieces": max((item["short_pieces"] for item in rows if item["short_pieces"] is not None), default=None),
        }
    machine = demo.SAL_INSTALLATION.default_machine
    return {"python": platform.python_version(), "platform": platform.platform(),
            "graph": demo.demo_config.graph_types[demo.graph_type_idx],
            "min_piece_factor": demo.min_piece_factor,
            "diameters_mm": [machine.small_duct_diameter_mm, machine.large_duct_diameter_mm],
            "repeats": repeats, "summary": summaries, "samples": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1:
        parser.error("cases and repeats must be positive")
    result = benchmark(args.cases, args.repeats)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
