"""Compare Cli directed-Steiner baseline and hard lengths at identical poses.

Timings cover the Cli scene solve (graph, candidates, search and sizing), but
exclude dwelling preparation, automatic placement, rendering and history.
"""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
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
    demo.installation_header_state = replace(
        demo.installation_header_state, enabled=frozenset({"cli"}), active="cli",
    )
    state = demo.workbench_state.cli
    state.room_start_mode_idx = 1
    state.return_ventilation_enabled = False
    samples = []
    for case in range(min(cases, len(demo.REAL_DWELLING_SCENARIOS) or 1)):
        demo.real_scenario_idx = case
        with redirect_stdout(io.StringIO()):
            demo.dwelling_runtime.generate()
        pose = demo.workbench_state.installation("cli").pose
        x, y, angle = pose.as_tuple()
        state.auto_placement_mode_idx = 0
        for dx, dy, rotation in ((0, 0, 0), (100, 0, 0), (-100, 0, 0),
                                 (0, 100, 0), (0, -100, 0), (0, 0, 90)):
            pose.set((x + dx, y + dy), angle + rotation)
            for backend in (4, 5, 6):
                state.router_backend_idx = backend
                with redirect_stdout(io.StringIO()):
                    demo.routing_controller.solve_installations()
                for _ in range(repeats):
                    with redirect_stdout(io.StringIO()):
                        started = perf_counter()
                        demo.routing_controller.solve_installations()
                        scene_ms = (perf_counter() - started) * 1000
                    result = demo.cli_runtime.last_result
                    stats = result.supply_length_stats
                    samples.append({
                        "case": demo.current_scenario_label, "pose": pose.as_tuple(),
                        "backend": backend, "scene_ms": scene_ms,
                        "solver_ms": result.elapsed_ms, "success": bool(result.routes),
                        "status": result.status, "nodes": result.total_nodes,
                        "minimum_mm": result.supply_minimum_length_mm,
                        "states": stats.state_count if stats else None,
                        "subset_labels": stats.subset_labels if stats else None,
                        "branch_labels": stats.branch_labels if stats else None,
                        "rejected_tree": stats.rejected_physical_tree if stats else False,
                    })
        state.auto_placement_mode_idx = 1
    summary = {}
    for backend in (4, 5, 6):
        rows = [row for row in samples if row["backend"] == backend]
        times = sorted(row["scene_ms"] for row in rows)
        summary[demo.CLI_INSTALLATION.search_backend_labels[backend]] = {
            "samples": len(rows), "successful": sum(row["success"] for row in rows),
            "median_ms": statistics.median(times),
            "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))],
            "max_ms": max(times), "over_200_ms": sum(value > 200 for value in times),
            "max_states": max((row["states"] or 0 for row in rows), default=0),
            "rejected_trees": sum(row["rejected_tree"] for row in rows),
            "max_subset_labels": max((row["subset_labels"] or 0 for row in rows), default=0),
            "max_branch_labels": max((row["branch_labels"] or 0 for row in rows), default=0),
        }
    return {"python": platform.python_version(), "graph": state.graph_type_idx,
            "machine": demo.installation_ui_session.selected_keys["cli"],
            "repeats": repeats, "summary": summary, "samples": samples}


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
