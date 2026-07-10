# Config Taxonomy Reference

Use semantic keys:

```text
<SCOPE>.<AREA>.<SUBAREA>.<PARAMETER>
```

Name values by why they exist, not by the function that consumes them.

## Current Decisions

- Use English keys.
- Use integer millimetres internally wherever practical.
- UI may display metres.
- Do not use `REGULATION` without an explicit legal/code/project-code source.
- Current Demo 10.8.1 distances are mostly `FEASIBILITY`, `HEURISTIC`, `SOLVER`, or `PERFORMANCE`.

## Classification

- `FEASIBILITY`: breaking it makes geometry invalid or impossible.
- `SOLVER`: changing it changes graph/search/cost algorithm behavior.
- `HEURISTIC`: professional preference or warning/scoring behavior, not validity.
- `MAINTENANCE`: serviceability/access after installation.
- `DEBUG`: inspection-only execution help.
- `OBSERVABILITY`: logs, traces, config snapshots.
- `PATHS`: directories, templates, families, logs.
- `REGULATION`: only confirmed legal/code/project-code constraints.

## Examples

Prefer:

```text
SALUBRIDAD.FEASIBILITY.ROUTING.MIN_DISTANCE_TO_WALL_MM
SALUBRIDAD.SOLVER.GRAPH.GRID_EPSILON_MM
SALUBRIDAD.HEURISTIC.DUCTS.MIN_PIECE_FACTOR_DEFAULT
DOMAIN.GEOMETRY.WALL_THICKNESS_MM
```

Avoid:

```text
SALUBRIDAD.SOLVER.HEURISTIC.BOCAS_MIN_DISTANCE_TO_WALL
SALUBRIDAD.REGULATION.*  # unless sourced
```

