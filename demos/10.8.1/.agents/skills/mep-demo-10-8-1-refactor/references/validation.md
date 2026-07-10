# Validation Reference

Use the cheapest validation that can catch likely regressions for the current step.

## Always

Run `python -m py_compile` on changed Python files.

## Pure Modules

Add or run focused tests when functions are extracted and can be imported without launching Pygame.

Good targets:

- config key resolution
- unit conversion
- machine pin geometry
- route scoring
- clearance classification
- graph builder invariants

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

