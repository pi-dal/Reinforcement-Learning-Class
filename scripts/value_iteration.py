from __future__ import annotations

from typing import Iterable

import numpy

from .utils import GridWorld, GridWorldSpec, closed_form_value

DEFAULT_SPEC = GridWorldSpec(H=5, W=5, goal=(4, 4))
TARGET_GAMMAS: Iterable[float] = (0.0, 0.5, 0.9)


def _format_value_grid(world: GridWorld, values: numpy.ndarray) -> str:
    grid = world.gridify(values)
    lines = []
    for row in grid:
        pieces = [f"{val:8.3f}" for val in row]
        lines.append(" ".join(pieces))
    return "\n".join(lines)


def _format_policy(world: GridWorld, policy) -> str:
    grid = world.policy_arrows(policy)
    lines = []
    for row in grid:
        lines.append(" ".join(f"{cell:>3}" if cell else "  ." for cell in row))
    return "\n".join(lines)


def run_one(gamma: float, spec: GridWorldSpec = DEFAULT_SPEC):
    world = GridWorld(spec)
    values, policy = world.value_iteration(gamma)
    try:
        closed = closed_form_value(spec, gamma)
        max_abs_err = float(numpy.max(numpy.abs(values - closed)))
    except ValueError:
        closed = None
        max_abs_err = None
    return world, values, policy, closed, max_abs_err


def main() -> None:
    for gamma in TARGET_GAMMAS:
        world, values, policy, closed, err = run_one(gamma)
        print(f"\n=== gamma = {gamma} ===")
        print("Value function (V*):")
        print(_format_value_grid(world, values))
        print("\nGreedy policy (arrows):")
        print(_format_policy(world, policy))
        if err is not None:
            print(f"\nmax |V* - closed-form| = {err:.3e}")
        else:
            print("\nclosed-form baseline unavailable for this configuration")


if __name__ == "__main__":
    main()
