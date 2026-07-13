"""Compare Clima routing backends on the configured real dwelling scenarios.

This is a measurement harness, not a replacement solver.  It exercises the
same placement, graph construction, and routing functions used by main.py.
"""

import argparse
import contextlib
import csv
import io
import os
import sys
from pathlib import Path

# Keep imports headless: main.py imports pygame but creates its window only when
# run as a program.  The dummy drivers also make this script usable in CI.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import main


FIELDNAMES = [
    "scenario_index",
    "scenario",
    "routing_frame",
    "machine",
    "graph",
    "backend",
    "status",
    "elapsed_ms",
    "steiner_tree_nodes",
    "route_length_m",
    "turns",
    "crossings",
    "overlaps",
    "clearance_conflicts",
    "experimental_short_pieces",
    "diagonal_segments",
    "outside_allowed_segments",
    "raw_tree_segments",
    "directed_tree_segments",
    "terminal_access_segments",
    "bridge_points",
    "connector_stub_segments",
    "validation_warnings",
]


def count_diagonal_segments(routes, tolerance=1e-6):
    if not routes:
        return 0
    return sum(
        1
        for _, segments in routes
        for p1, p2 in segments
        if abs(float(p2[0]) - float(p1[0])) > tolerance
        and abs(float(p2[1]) - float(p1[1])) > tolerance
    )


def build_row(scenario_index, routes, status, elapsed_ms, total_nodes):
    debug = main.core_steiner_debug
    warnings = main.get_route_validation_warnings(routes)
    return {
        "scenario_index": scenario_index,
        "scenario": main.current_scenario_label,
        "routing_frame": main.ROUTING_FRAME_OPTIONS[main.routing_frame_idx],
        "machine": main.get_current_machine()["label"],
        "graph": main.GRAPH_TYPES[main.graph_type_idx],
        "backend": main.CLIMA_SUPPLY_BACKENDS[main.clima_supply_backend_idx],
        "status": status,
        "elapsed_ms": round(float(elapsed_ms), 3),
        "steiner_tree_nodes": int(total_nodes),
        "route_length_m": round(main._routes_total_length_m(routes), 3),
        "turns": main.count_solution_turns(routes) if routes else 0,
        "crossings": main.count_segment_crossings(routes) if routes else 0,
        "overlaps": main.count_segment_overlaps(routes) if routes else 0,
        "clearance_conflicts": main.count_segment_clearance_conflicts(routes) if routes else 0,
        "experimental_short_pieces": main.count_solution_short_pieces(routes) if routes else 0,
        "diagonal_segments": count_diagonal_segments(routes),
        "outside_allowed_segments": len(main.get_route_outside_allowed_segments(routes)),
        "raw_tree_segments": len(debug.get("raw_tree_segments", [])),
        "directed_tree_segments": len(debug.get("directed_tree_segments", [])),
        "terminal_access_segments": len(debug.get("access_segments", [])),
        "bridge_points": len(debug.get("bridge_points", [])),
        "connector_stub_segments": len(debug.get("stub_segments", [])),
        "validation_warnings": "; ".join(warnings),
    }


def parse_indices(value, upper_bound, option_name):
    if value == "all":
        return list(range(upper_bound))
    try:
        indices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option_name} must be 'all' or comma-separated indices") from error
    invalid = [index for index in indices if index < 0 or index >= upper_bound]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid {option_name} index: {invalid[0]}")
    return indices


def run_comparison(scenario_indices, graph_indices, backend_indices, routing_frame_index):
    rows = []
    main.dwelling_source_idx = main.DWELLING_SOURCE_MODES.index("Real DB")
    main.auto_placement_mode_idx = main.AUTO_PLACEMENT_MODES.index("Routing-Core Workflow")
    main.routing_frame_idx = routing_frame_index

    for scenario_index in scenario_indices:
        for graph_index in graph_indices:
            main.real_scenario_idx = scenario_index
            main.graph_type_idx = graph_index
            with contextlib.redirect_stdout(io.StringIO()):
                main.generate_new_dwelling()
            for backend_index in backend_indices:
                main.clima_supply_backend_idx = backend_index
                with contextlib.redirect_stdout(io.StringIO()):
                    routes, status, elapsed_ms, total_nodes = main.solve_clima_routing()
                rows.append(build_row(scenario_index, routes, status, elapsed_ms, total_nodes))
    return rows


def write_rows(rows, output):
    if output is None:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="all", help="'all' or comma-separated real scenario indices")
    parser.add_argument("--graphs", default="all", help="'all' or comma-separated graph indices")
    parser.add_argument("--backends", default="all", help="'all' or comma-separated backend indices")
    parser.add_argument("--frame", type=int, default=0, choices=range(len(main.ROUTING_FRAME_OPTIONS)))
    parser.add_argument("--output", type=Path, help="optional CSV output path")
    args = parser.parse_args()

    scenarios = parse_indices(args.scenarios, len(main.REAL_DWELLING_SCENARIOS), "scenario")
    graphs = parse_indices(args.graphs, len(main.GRAPH_TYPES), "graph")
    backends = parse_indices(args.backends, len(main.CLIMA_SUPPLY_BACKENDS), "backend")
    rows = run_comparison(scenarios, graphs, backends, args.frame)
    write_rows(rows, args.output)


if __name__ == "__main__":
    main_cli()
