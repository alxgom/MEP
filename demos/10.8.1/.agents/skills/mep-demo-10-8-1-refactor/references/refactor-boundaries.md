# Refactor Boundaries Reference

Keep module ownership stable while extracting from `main.py`.

## Boundaries

- `config`: typed defaults, metadata, aliases, loading.
- `domain`: machine families, terminal families, room categories, shaft semantics, duct families, feasibility defaults, validation concepts.
- `geometry`: generic transforms and geometric helpers; no Sal/Cli/Coc/San branching.
- `graphs`: graph structures, graph builders, edge fields.
- `routing`: A*, line graph search, min-cost flow, strategy orchestration, scoring.
- `placement`: machine candidates, placement scoring, rotation scoring.
- `data_sources`: synthetic and dwelling export adapters.
- `ui`: Pygame drawing, layout, event handling, widgets, plots.
- `observability`: logs, snapshots, debug dumps.

## Extraction Order

1. Import-safe value objects.
2. Pure geometry helpers.
3. Graph environment and builders.
4. Routing backends.
5. Scoring and validation.
6. Placement.
7. Data sources.
8. UI drawing/events.

## Rules

- Do not move UI dependencies into pure modules.
- Do not let graph builders read global Pygame state.
- Pass context explicitly when practical.
- Keep temporary wrappers if they reduce migration risk.
- Avoid broad style cleanups while moving behavior.

