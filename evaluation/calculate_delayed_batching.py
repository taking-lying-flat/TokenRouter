from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix, eye, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve


State = tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ThresholdResult:
    """Exact-model result for one threshold vector."""

    B: tuple[int, ...]
    throughput: float
    num_states: int
    num_closed_classes: int


@dataclass(frozen=True)
class OptimizationResult:
    """Optimal vector and the result for every feasible vector."""

    optimal_B: tuple[int, ...]
    max_throughput: float
    time_unit: Fraction
    results: tuple[ThresholdResult, ...]


def _as_fraction(value: object) -> Fraction:
    """Convert decimal input without inheriting float binary error."""

    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def common_time_grid(
    latencies: Sequence[object], delta: object | None = None
) -> tuple[Fraction, tuple[int, ...]]:
    """Return the selected exact time unit and integer latency ticks.

    When ``delta`` is omitted, the greatest exact common unit is inferred.
    When supplied, every latency must be an integer multiple of ``delta``.
    """

    values = tuple(_as_fraction(value) for value in latencies)
    if not values or any(value <= 0 for value in values):
        raise ValueError("all latencies must be positive")
    if delta is not None:
        unit = _as_fraction(delta)
        if unit <= 0:
            raise ValueError("delta must be positive")
        ratios = tuple(value / unit for value in values)
        if any(ratio.denominator != 1 for ratio in ratios):
            raise ValueError(
                "every latency must be an exact integer multiple of delta; "
                f"got latencies={list(latencies)!r}, delta={delta!r}"
            )
        return unit, tuple(ratio.numerator for ratio in ratios)

    denominator = math.lcm(*(value.denominator for value in values))
    numerators = tuple(value.numerator * (denominator // value.denominator) for value in values)
    common_numerator = math.gcd(*numerators)
    unit = Fraction(common_numerator, denominator)
    return unit, tuple(numerator // common_numerator for numerator in numerators)


def _validate_inputs(
    M: int,
    latencies: Sequence[object],
    N: int,
    P: Sequence[Sequence[float]],
    C: Sequence[Sequence[float]],
    initial_counts: Sequence[int] | None,
    delta: object | None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    if M <= 0:
        raise ValueError("M must be positive")
    if N <= 0:
        raise ValueError("N must be positive")
    if len(latencies) != M:
        raise ValueError(f"latencies must contain M={M} values")
    common_time_grid(latencies, delta)

    probabilities = np.asarray(P, dtype=float)
    commits = np.asarray(C, dtype=float)
    if probabilities.shape != (M, M):
        raise ValueError(f"P must have shape ({M}, {M})")
    if commits.shape != (M, M):
        raise ValueError(f"C must have shape ({M}, {M})")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("P must contain finite, non-negative probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("every row of P must sum to 1")
    if not np.all(np.isfinite(commits)):
        raise ValueError("C must contain finite values")
    if np.any(commits < 0):
        raise ValueError("C must contain non-negative token counts")

    if initial_counts is None:
        initial = (N,) + (0,) * (M - 1)
    else:
        initial = tuple(initial_counts)
        if len(initial) != M or any(
            not isinstance(x, (int, np.integer)) or x < 0 for x in initial
        ):
            raise ValueError("initial_counts must contain M non-negative integers")
        if sum(initial) != N:
            raise ValueError("initial_counts must sum to N")
    return probabilities, commits, initial


def feasible_thresholds(M: int, N: int) -> Iterable[tuple[int, ...]]:
    """Yield B in lexicographic order with sum(B_i - 1) < N."""

    if M <= 0 or N <= 0:
        return

    prefix: list[int] = []

    def visit(position: int, remaining_excess: int):
        if position == M:
            yield tuple(prefix)
            return
        for excess in range(remaining_excess + 1):
            prefix.append(excess + 1)
            yield from visit(position + 1, remaining_excess - excess)
            prefix.pop()

    # Strict inequality means the total excess can be at most N - 1.
    yield from visit(0, N - 1)


def _launch(waiting: Sequence[int], B: tuple[int, ...], ticks: tuple[int, ...]) -> State:
    state = []
    for i, count in enumerate(waiting):
        if count >= B[i]:
            state.append((0, count, ticks[i]))
        else:
            state.append((count, 0, 0))
    return tuple(state)


@lru_cache(maxsize=None)
def _multinomial_outcomes(batch_size: int, probabilities: tuple[float, ...]):
    """Enumerate positive-probability destination counts and probabilities."""

    support = tuple(i for i, probability in enumerate(probabilities) if probability > 0.0)
    outcomes: list[tuple[tuple[int, ...], float]] = []
    counts = [0] * len(probabilities)

    def visit(position: int, remaining: int) -> None:
        if position == len(support) - 1:
            counts[support[position]] = remaining
            coefficient = math.factorial(batch_size)
            probability = 1.0
            for j, count in enumerate(counts):
                coefficient //= math.factorial(count)
                probability *= probabilities[j] ** count
            outcomes.append((tuple(counts), coefficient * probability))
            counts[support[position]] = 0
            return
        destination = support[position]
        for count in range(remaining + 1):
            counts[destination] = count
            visit(position + 1, remaining - count)
        counts[destination] = 0

    visit(0, batch_size)
    return tuple(outcomes)


def _state_transitions(
    state: State,
    B: tuple[int, ...],
    ticks: tuple[int, ...],
    probabilities: np.ndarray,
) -> tuple[dict[State, float], int, tuple[int, ...]]:
    running = tuple(i for i, (_, batch, _) in enumerate(state) if batch > 0)
    if not running:
        raise RuntimeError("deadlocked state encountered; threshold domain should exclude it")
    tau = min(state[i][2] for i in running)
    active = tuple(i for i in running if state[i][2] == tau)
    active_set = set(active)
    per_source = [
        _multinomial_outcomes(state[i][1], tuple(float(x) for x in probabilities[i]))
        for i in active
    ]
    transitions: dict[State, float] = defaultdict(float)
    for joint_outcome in product(*per_source):
        arrivals = [0] * len(state)
        branch_probability = 1.0
        for counts, probability in joint_outcome:
            branch_probability *= probability
            for j, count in enumerate(counts):
                arrivals[j] += count

        updated = []
        for j, (waiting, batch, remaining) in enumerate(state):
            next_waiting = waiting + arrivals[j]
            if j in active_set:
                next_batch, next_remaining = 0, 0
            else:
                next_batch = batch
                next_remaining = remaining - tau if batch else 0
            if next_batch == 0 and next_waiting >= B[j]:
                updated.append((0, next_waiting, ticks[j]))
            else:
                updated.append((next_waiting, next_batch, next_remaining))
        transitions[tuple(updated)] += branch_probability
    return dict(transitions), tau, active


def _enumerate_chain(
    initial_state: State,
    B: tuple[int, ...],
    ticks: tuple[int, ...],
    probabilities: np.ndarray,
    commits: np.ndarray,
    max_states: int | None,
):
    states = [initial_state]
    state_to_index = {initial_state: 0}
    rows: list[dict[int, float]] = []
    holding_ticks: list[int] = []
    rewards: list[float] = []
    cursor = 0
    while cursor < len(states):
        state = states[cursor]
        next_states, tau, active = _state_transitions(state, B, ticks, probabilities)
        row: dict[int, float] = defaultdict(float)
        for next_state, probability in next_states.items():
            if next_state not in state_to_index:
                if max_states is not None and len(states) >= max_states:
                    raise RuntimeError(
                        f"reachable state count exceeded max_states={max_states} for B={B}"
                    )
                state_to_index[next_state] = len(states)
                states.append(next_state)
            row[state_to_index[next_state]] += probability
        total = sum(row.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"transition probabilities sum to {total}, not 1")
        rows.append(dict(row))
        holding_ticks.append(tau)
        rewards.append(
            sum(
                state[i][1] * float(np.dot(probabilities[i], commits[i]))
                for i in active
            )
        )
        cursor += 1

    size = len(states)
    matrix = lil_matrix((size, size), dtype=float)
    for i, row in enumerate(rows):
        for j, probability in row.items():
            matrix[i, j] = probability
    return states, matrix.tocsr(), np.asarray(holding_ticks, dtype=float), np.asarray(rewards)


def _strongly_connected_components(matrix: csr_matrix) -> list[list[int]]:
    """Return SCCs over positive-probability edges without Python recursion."""

    matrix = matrix.copy()
    matrix.eliminate_zeros()
    count, labels = connected_components(
        matrix, directed=True, connection="strong", return_labels=True
    )
    components = [[] for _ in range(count)]
    for state, label in enumerate(labels):
        components[int(label)].append(state)
    return components


def _closed_components(matrix: csr_matrix) -> list[list[int]]:
    """Return all closed communicating classes of a finite Markov chain."""

    components = _strongly_connected_components(matrix)
    component_of = {
        state: component_index
        for component_index, component in enumerate(components)
        for state in component
    }
    closed = []
    for component_index, component in enumerate(components):
        if all(
            all(
                component_of[int(destination)] == component_index
                for destination in matrix.indices[
                    matrix.indptr[state] : matrix.indptr[state + 1]
                ]
                if matrix[state, destination] != 0.0
            )
            for state in component
        ):
            closed.append(sorted(component))
    return closed


def stationary_solution_is_unique(matrix: csr_matrix) -> bool:
    """Check uniqueness before solving ``eta K = eta, sum(eta) = 1``.

    A finite Markov chain has a unique stationary distribution exactly when
    its transition graph contains one closed communicating class. Transient
    states are allowed and receive zero stationary mass.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("transition matrix must be non-empty and square")
    return len(_closed_components(matrix)) == 1


def _stationary_distribution(matrix: csr_matrix) -> np.ndarray:
    """Solve eta K = eta with one equation replaced by normalization."""

    size = matrix.shape[0]
    closed_count = len(_closed_components(matrix))
    if closed_count != 1:
        raise ValueError(
            "stationary equation does not have a unique solution: "
            f"found {closed_count} closed communicating classes"
        )
    if size == 1:
        return np.ones(1)
    system = (matrix.T - eye(size, format="csr")).tolil()
    system[-1, :] = np.ones(size)
    rhs = np.zeros(size)
    rhs[-1] = 1.0
    stationary = np.asarray(spsolve(system.tocsr(), rhs), dtype=float)
    stationary[np.abs(stationary) < 1e-14] = 0.0
    if np.any(stationary < -1e-10) or not np.all(np.isfinite(stationary)):
        raise RuntimeError("failed to compute a valid stationary distribution")
    stationary = np.maximum(stationary, 0.0)
    return stationary / stationary.sum()


def _long_run_throughput(
    matrix: csr_matrix,
    holding_ticks: np.ndarray,
    rewards: np.ndarray,
    time_unit: Fraction,
) -> tuple[float, int]:
    closed = _closed_components(matrix)

    class_throughputs = []
    for component in closed:
        submatrix = matrix[component, :][:, component].tocsr()
        stationary = _stationary_distribution(submatrix)
        numerator = float(stationary @ rewards[component])
        denominator = float(stationary @ holding_ticks[component]) * float(time_unit)
        class_throughputs.append(numerator / denominator)

    initial_class = next((i for i, component in enumerate(closed) if 0 in component), None)
    if initial_class is not None:
        return class_throughputs[initial_class], len(closed)

    closed_states = {state for component in closed for state in component}
    transient = [state for state in range(matrix.shape[0]) if state not in closed_states]
    transient_position = {state: i for i, state in enumerate(transient)}
    q = matrix[transient, :][:, transient].tocsr()
    direct = np.zeros((len(transient), len(closed)))
    class_of_state = {
        state: class_index for class_index, component in enumerate(closed) for state in component
    }
    for state in transient:
        row = transient_position[state]
        start, end = matrix.indptr[state], matrix.indptr[state + 1]
        for destination, probability in zip(matrix.indices[start:end], matrix.data[start:end]):
            destination = int(destination)
            if destination in class_of_state:
                direct[row, class_of_state[destination]] += probability
    absorption = np.asarray(spsolve(eye(len(transient), format="csr") - q, direct))
    if absorption.ndim == 1:
        absorption = absorption[:, None]
    weights = absorption[transient_position[0]]
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum()
    return float(weights @ np.asarray(class_throughputs)), len(closed)


def solve_thresholds(
    B: Sequence[int],
    M: int,
    latencies: Sequence[object],
    N: int,
    P: Sequence[Sequence[float]],
    C: Sequence[Sequence[float]],
    *,
    initial_counts: Sequence[int] | None = None,
    delta: object | None = None,
    max_states: int | None = 100_000,
) -> ThresholdResult:
    """Compute throughput for one B vector using its reachable event chain."""

    probabilities, commits, initial = _validate_inputs(
        M, latencies, N, P, C, initial_counts, delta
    )
    time_unit, ticks = common_time_grid(latencies, delta)
    thresholds = tuple(B)
    if len(thresholds) != M or any(
        not isinstance(x, (int, np.integer)) or not 1 <= x <= N for x in thresholds
    ):
        raise ValueError("B must contain M integers in [1, N]")
    if sum(value - 1 for value in thresholds) >= N:
        raise ValueError("B must satisfy sum(B_i - 1) < N to exclude global deadlock")

    initial_state = _launch(initial, thresholds, ticks)
    states, matrix, holding_ticks, rewards = _enumerate_chain(
        initial_state, thresholds, ticks, probabilities, commits, max_states
    )
    throughput, class_count = _long_run_throughput(
        matrix, holding_ticks, rewards, time_unit
    )
    return ThresholdResult(thresholds, throughput, len(states), class_count)


def optimize_batch_thresholds(
    M: int,
    latencies: Sequence[object],
    N: int,
    P: Sequence[Sequence[float]],
    C: Sequence[Sequence[float]],
    *,
    initial_counts: Sequence[int] | None = None,
    delta: object | None = None,
    max_states: int | None = 100_000,
) -> OptimizationResult:
    """Enumerate the deadlock-free domain and return the optimal B vector."""

    _validate_inputs(M, latencies, N, P, C, initial_counts, delta)
    time_unit, _ = common_time_grid(latencies, delta)
    results = tuple(
        solve_thresholds(
            B,
            M,
            latencies,
            N,
            P,
            C,
            initial_counts=initial_counts,
            delta=delta,
            max_states=max_states,
        )
        for B in feasible_thresholds(M, N)
    )
    best = max(results, key=lambda result: result.throughput)
    return OptimizationResult(best.B, best.throughput, time_unit, results)


def calculate_optimal_B(
    M: int,
    L: Sequence[object],
    N: int,
    P: Sequence[Sequence[float]],
    C: Sequence[Sequence[float]],
    **kwargs,
) -> tuple[int, ...]:
    """Convenience API returning only the optimal threshold vector."""

    return optimize_batch_thresholds(M, L, N, P, C, **kwargs).optimal_B


def _json_value(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("config JSON must be an object")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON file containing M, N, latencies, P and C")
    parser.add_argument("--M", type=int, help="number of models")
    parser.add_argument("--N", type=int, help="fixed request concurrency")
    parser.add_argument(
        "--latencies", "--L", dest="latencies", nargs="+", help="M fixed per-step latencies"
    )
    parser.add_argument("--P", type=_json_value, help="M x M routing probability matrix as JSON")
    parser.add_argument("--C", type=_json_value, help="M x M token-commit matrix as JSON")
    parser.add_argument("--initial-counts", nargs="+", type=int)
    parser.add_argument(
        "--delta",
        help="common time unit; inferred exactly when omitted (for example: 1 or 0.1)",
    )
    parser.add_argument("--max-states", type=int, default=100_000)
    parser.add_argument(
        "--latency-unit",
        choices=("s", "ms", "us"),
        default="ms",
        help="latency unit; controls only displayed token/s (default: ms)",
    )
    parser.add_argument("--show-all", action="store_true", help="print every feasible B")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config) if args.config else {}
    if "latencies" not in config and "L" in config:
        config["latencies"] = config["L"]
    supplied = {
        "M": args.M,
        "N": args.N,
        "latencies": args.latencies,
        "P": args.P,
        "C": args.C,
        "initial_counts": args.initial_counts,
        "delta": args.delta,
    }
    for key, value in supplied.items():
        if value is not None:
            config[key] = value
    missing = [key for key in ("M", "N", "latencies", "P", "C") if key not in config]
    if missing:
        raise ValueError("missing required inputs: " + ", ".join(missing))

    result = optimize_batch_thresholds(
        config["M"],
        config["latencies"],
        config["N"],
        config["P"],
        config["C"],
        initial_counts=config.get("initial_counts"),
        delta=config.get("delta"),
        max_states=args.max_states,
    )
    per_second = {"s": 1.0, "ms": 1000.0, "us": 1_000_000.0}[args.latency_unit]
    if args.show_all:
        for item in result.results:
            print(
                f"B={list(item.B)}, throughput={item.throughput * per_second:.9g} token/s, "
                f"states={item.num_states}, closed_classes={item.num_closed_classes}"
            )
    print(f"common time unit = {result.time_unit} {args.latency_unit}")
    print(f"optimal B = {list(result.optimal_B)}")
    print(f"max throughput = {result.max_throughput * per_second:.9g} token/s")


if __name__ == "__main__":
    main()
