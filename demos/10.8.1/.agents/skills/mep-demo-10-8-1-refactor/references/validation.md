# Validation Reference

Use the cheapest validation that can catch likely regressions for the current step. Scale validation to behavioral risk, not the number of functions moved.

## Always

Run `python -m py_compile` on changed Python files.

## Contract Tests

Add or run focused tests only when an extraction introduces or changes a reusable public contract, solver invariant, geometry rule, or configuration behavior.

Do not add tests solely for forwarding wrappers, unchanged formatting, or Pygame drawing relocation. Compile those changes and validate them at the applicable manual milestone.

Good targets:

- config key resolution
- unit conversion
- machine pin geometry
- route scoring
- clearance classification
- graph builder invariants
- routing strategy outcomes for representative scenarios

## Manual Milestones

Run the Pygame app manually after:

- first entrypoint split
- graph builder extraction
- routing strategy extraction
- placement extraction
- UI event extraction
- config runtime consumption
- moving the app outside `demos/10.8.1`

Manual checks:

- app launches
- real and synthetic dwelling modes still work where available
- graph mode cycling still works
- solver strategy cycling still works
- machine drag/rotate works
- terminal point/area tools still work
- route selection still works
- plots/logging still update
