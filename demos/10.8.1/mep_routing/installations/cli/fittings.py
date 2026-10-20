"""Numeric centerline fitting counts for the Cli routing experiments.

Each elbow, tee or cross counts as one component. Collinear size changes,
straight couplings and terminal/machine connections are outside this lab model.
"""

from dataclasses import dataclass
import math


FITTING_PERTURBATION_MM = 1e-6


@dataclass(frozen=True)
class FittingCounts:
    elbows: int = 0
    tees: int = 0
    crosses: int = 0

    @property
    def total(self):
        return self.elbows + self.tees + self.crosses


def count_tree_fittings(lines, terminal_points=()):
    """Count physical junctions once, including zero-length grille takeoffs.

    Input is the validated physical tree. Incident rays merge collinear grid
    and materialized size splits; zero-length graph aliases add no ports.
    """
    ports = {}
    terminals = {tuple(map(float, point[:2])) for point in terminal_points}
    for start, end in lines:
        start, end = tuple(map(float, start[:2])), tuple(map(float, end[:2]))
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-7:
            continue
        direction = round(dx / length, 9), round(dy / length, 9)
        ports.setdefault(start, set()).add(direction)
        ports.setdefault(end, set()).add(tuple(-value for value in direction))
    elbows = tees = crosses = 0
    for point, directions in ports.items():
        count = len(directions)
        if count < 2:
            continue
        if point in terminals:
            count += 1
        if count >= 4:
            crosses += 1
        elif count == 3:
            tees += 1
        else:
            first, second = directions
            if first != tuple(-value for value in second):
                elbows += 1
    return FittingCounts(elbows, tees, crosses)


def junction_fitting_count(occupied, incoming_ray, straight_masks):
    """Classify existing DP port masks; bit zero denotes a local grille end."""
    ports = (occupied & ~1) | incoming_ray
    count = ports.bit_count()
    return int(count > 2 or (count == 2 and (occupied & 1 or ports not in straight_masks)))
