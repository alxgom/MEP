# Interactive Platform Guidelines

Use Demo 10.8.1 to reproduce comparable `letsmep-routing-core` behavior while developing and testing faster implementations and new ideas.

Treat these as defaults, not fixed architecture. The platform is intentionally mutable: revise a pattern when an experiment, measurement, or clearer boundary supports doing so. Keep deviations visible enough to understand and compare.

## Comparison Contract

When useful for comparison, capture:

- routing-core source class/function and relevant configuration;
- comparable inputs, units, connector directions, allowed regions, obstacles, and terminal semantics;
- observable intermediate artifacts and output invariants;
- intentional implementation and behavioral deviations;
- latency, memory, determinism, route-quality, and failure-mode effects when material.

It can help to describe work as a faithful port, adapted port, experiment, or demo-only tooling. Similar final geometry is useful evidence, but not sufficient by itself to establish parity.

## Execution Priorities

- Prefer NumPy arrays, vectorized geometry, compact integer indices, and spatial indexes on repeated paths when they improve interactive performance or clarity.
- Keep coordinates and dimensions as integer millimetres where practical. Convert only at input, display, or external API boundaries.
- Use Shapely freely where it is the clearest reliable tool, especially for normalization and topology-heavy preparation. Consider tested numeric alternatives on measured hot paths.
- Build immutable dwelling/base graph geometry once. Reuse nodes, edges, adjacency, axes, and spatial indexes across interactive changes.
- Represent machine, route, and ordered-installation context as masks, exclusions, or node/edge cost layers when topology need not change.
- Prefer explicit cache owners and geometry-sensitive keys. Invalidate the narrowest affected layer that can be identified safely.
- Prefer diagnostics that project existing artifacts and do not alter solving. Accept additional diagnostic work when its cost is explicit and useful.
- Keep pure computation independent from Pygame, filesystem access, and mutable UI globals.

## Behavioral Considerations

- Preserve physical connector points, directions, access offsets, and final stubs separately from snapped or adjusted graph terminals.
- Avoid invented representative-point or diagonal fallback terminals merely to produce a route; exposing disconnected or rejected terminals usually provides better experimental evidence.
- Prefer deterministic ordering and tie-breaking so identical integer inputs reproduce identical results.
- Choose deliberately between high-cost edges, masks, and hard removal. Physically impossible traversal, such as crossing a machine body, normally favors removal; routing-core comparison or recovery experiments may justify retaining a costly edge.
- Treat connector direction as a hard constraint only when the compared behavior requires it and the graph supports it. Otherwise use an explicit finite penalty and document the deviation.
- Prefer keeping validation separate from solving. If an experiment couples validation to retry, relaxation, or ranking, expose that policy explicitly.
- Make any context relaxation or fallback explicit in the result status and diagnostics.
- Preserve invalid experimental outcomes when they are useful evidence; do not hide them by automatically selecting a different candidate.

## Experiment Design

Prefer changing one coherent mechanism at a time and retaining a baseline backend or configuration where feasible. Comparable dwellings and machine poses make differences easier to interpret. Capture enough artifacts to explain important differences, not only total score or runtime.

Before proposing upstream adoption, distinguish domain behavior from platform-specific acceleration. Pygame state, visualization contracts, and demo cache plumbing usually need not move upstream with an algorithmic improvement.
