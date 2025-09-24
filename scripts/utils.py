from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy

Action = int
StateIndex = int
Transition = Tuple[float, StateIndex, float, bool]

ACTION_DELTAS: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_SYMBOLS: Tuple[str, ...] = ("↑", "→", "↓", "←")


@dataclass
class GridWorldSpec:
    H: int
    W: int
    goal: Tuple[int, int]
    walls: Optional[Sequence[Tuple[int, int]]] = None
    step_cost: float = -1.0
    goal_reward: float = 10.0
    stay_on_wall: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.H <= 0 or self.W <= 0:
            raise ValueError("Grid dimensions must be positive")
        gi, gj = self.goal
        if not (0 <= gi < self.H and 0 <= gj < self.W):
            raise ValueError("Goal must lie inside the grid")
        walls = set(self.walls or [])
        if len(walls) != len(self.walls or []):
            raise ValueError("Duplicate wall coordinates are not allowed")
        for wi, wj in walls:
            if not (0 <= wi < self.H and 0 <= wj < self.W):
                raise ValueError("Wall coordinates must be inside the grid")
        if self.goal in walls:
            raise ValueError("Goal cannot coincide with a wall")
        if not self.stay_on_wall:
            raise ValueError("Only stay_on_wall=True is supported in this implementation")


class GridWorld:
    def __init__(self, spec: GridWorldSpec) -> None:
        self.spec = spec
        self._walls = set(spec.walls or [])
        self._coords: List[Tuple[int, int]] = []
        self._coord_to_index: Dict[Tuple[int, int], StateIndex] = {}
        for i in range(spec.H):
            for j in range(spec.W):
                if (i, j) in self._walls:
                    continue
                idx = len(self._coords)
                self._coords.append((i, j))
                self._coord_to_index[(i, j)] = idx
        self.goal_index = self._coord_to_index[spec.goal]
        self.n_states = len(self._coords)
        self.n_actions = len(ACTION_DELTAS)
        self._transitions: List[List[List[Transition]]] = [
            [list() for _ in range(self.n_actions)] for _ in range(self.n_states)
        ]
        self._build_transitions()

    def index_of(self, coord: Tuple[int, int]) -> StateIndex:
        return self._coord_to_index[coord]

    @property
    def coordinates(self) -> List[Tuple[int, int]]:
        return list(self._coords)

    def _build_transitions(self) -> None:
        spec = self.spec
        for s_idx, (i, j) in enumerate(self._coords):
            if s_idx == self.goal_index:
                for a_idx in range(self.n_actions):
                    self._transitions[s_idx][a_idx] = [(1.0, s_idx, 0.0, True)]
                continue
            for a_idx, (di, dj) in enumerate(ACTION_DELTAS):
                ni, nj = i + di, j + dj
                if not (0 <= ni < spec.H and 0 <= nj < spec.W) or (ni, nj) in self._walls:
                    ni, nj = i, j
                ns_idx = self._coord_to_index[(ni, nj)]
                done = ns_idx == self.goal_index
                reward = spec.goal_reward if done else spec.step_cost
                self._transitions[s_idx][a_idx] = [(1.0, ns_idx, reward, done)]

    def value_iteration(
        self, gamma: float, tol: float = 1e-8, max_iters: int = 10_000
    ) -> Tuple[numpy.ndarray, List[Optional[Action]]]:
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("Gamma must lie in [0, 1]")
        V = numpy.zeros(self.n_states, dtype=float)
        policy: List[Optional[Action]] = [None] * self.n_states
        for _ in range(max_iters):
            V_new = numpy.empty_like(V)
            for s_idx in range(self.n_states):
                best_action: Optional[Action] = None
                best_value = -numpy.inf
                for a_idx in range(self.n_actions):
                    total = 0.0
                    for prob, ns_idx, reward, done in self._transitions[s_idx][a_idx]:
                        continuation = 0.0 if done else gamma * V[ns_idx]
                        total += prob * (reward + continuation)
                    if total > best_value + 1e-12 or (
                        numpy.isclose(total, best_value, atol=1e-12) and best_action is not None and a_idx < best_action
                    ):
                        best_value = total
                        best_action = a_idx
                if best_action is None:
                    best_action = 0
                policy[s_idx] = None if s_idx == self.goal_index else best_action
                V_new[s_idx] = best_value
            delta = float(numpy.max(numpy.abs(V_new - V)))
            V[:] = V_new
            if delta < tol:
                break
        else:
            raise RuntimeError("Value iteration did not converge within max_iters")
        for s_idx in range(self.n_states):
            if s_idx == self.goal_index:
                policy[s_idx] = None
                continue
            q_best = -numpy.inf
            best_action: Optional[Action] = None
            for a_idx in range(self.n_actions):
                total = 0.0
                for prob, ns_idx, reward, done in self._transitions[s_idx][a_idx]:
                    continuation = 0.0 if done else gamma * V[ns_idx]
                    total += prob * (reward + continuation)
                if total > q_best + 1e-12 or (
                    numpy.isclose(total, q_best, atol=1e-12) and best_action is not None and a_idx < best_action
                ):
                    q_best = total
                    best_action = a_idx
            policy[s_idx] = best_action
        return V, policy

    def gridify(self, values: numpy.ndarray, *, fill_val: float = numpy.nan) -> numpy.ndarray:
        if values.shape != (self.n_states,):
            raise ValueError("values must have shape (n_states,)")
        grid = numpy.full((self.spec.H, self.spec.W), fill_val, dtype=float)
        for idx, (i, j) in enumerate(self._coords):
            grid[i, j] = float(values[idx])
        return grid

    def policy_arrows(self, policy: Sequence[Optional[Action]]) -> List[List[str]]:
        if len(policy) != self.n_states:
            raise ValueError("policy length must match number of states")
        arrows: List[List[str]] = [["" for _ in range(self.spec.W)] for _ in range(self.spec.H)]
        for idx, (i, j) in enumerate(self._coords):
            if (i, j) == self.spec.goal:
                arrows[i][j] = "G"
            else:
                action = policy[idx]
                arrows[i][j] = ACTION_SYMBOLS[action] if action is not None else ""
        for (i, j) in self._walls:
            arrows[i][j] = "#"
        return arrows


def closed_form_value(spec: GridWorldSpec, gamma: float) -> numpy.ndarray:
    if spec.walls:
        raise ValueError("closed_form_value only supports grids without walls")
    if spec.step_cost != -1.0 or spec.goal_reward != 10.0:
        raise ValueError("closed_form_value assumes step_cost=-1 and goal_reward=10")
    if not spec.stay_on_wall:
        raise ValueError("closed_form_value assumes stay_on_wall=True")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("Gamma must lie in [0, 1]")
    world = GridWorld(spec)
    goal = spec.goal
    values = numpy.zeros(world.n_states, dtype=float)
    for idx, (i, j) in enumerate(world.coordinates):
        if (i, j) == goal:
            values[idx] = 0.0
            continue
        distance = abs(goal[0] - i) + abs(goal[1] - j)
        if distance <= 0:
            values[idx] = 0.0
            continue
        if gamma == 0.0:
            values[idx] = spec.goal_reward if distance == 1 else spec.step_cost
            continue
        if gamma == 1.0:
            values[idx] = spec.goal_reward + (distance - 1) * spec.step_cost
            continue
        total = 0.0
        if distance > 1:
            total += spec.step_cost * (1 - gamma ** (distance - 1)) / (1 - gamma)
        total += (gamma ** (distance - 1)) * spec.goal_reward
        values[idx] = total
    return values
