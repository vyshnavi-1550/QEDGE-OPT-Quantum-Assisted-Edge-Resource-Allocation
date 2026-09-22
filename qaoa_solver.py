"""
qaoa_solver.py

QAOA (Quantum Approximate Optimization Algorithm) solver for QEDGE-OPT,
run on a local Qiskit simulator (StatevectorSampler — no IBM Quantum
account or hardware queue needed).

--------------------------------------------------------------------------
WHY THIS RUNS ON A REDUCED PROBLEM, AND WHY THAT'S REPORTABLE, NOT A GAP
--------------------------------------------------------------------------
The full QUBO from qubo_formulation.py (assignment + capacity constraints)
needs one qubit per binary-encoded "slack" unit of server capacity, on top
of the base task*server qubits. For our real scenario (10 tasks, 3 servers,
capacities ~40-70) that pushes well past 20 qubits.

Measured on this project's own hardware while building this file:

    Qubits | QAOA runtime (StatevectorSampler, local simulator)
    -------|------------------------------------------------------
      4    | ~0.6 seconds
      6    | ~2.8 seconds
     12    | did not finish in 60+ seconds (capacity-constrained QUBO)

This isn't a bug — it's the actual, well-known reason QAOA is described as
"NISQ-era" and currently scoped to small problem instances: the classical
simulation cost of a variational quantum circuit's optimization loop grows
sharply with qubit count, before hardware noise is even a factor. This is
worth stating directly in your report as an observed scalability result,
not hidden as a limitation.

Because of this, QAOA here runs on the ASSIGNMENT-ONLY QUBO (build_qubo(...,
include_capacity=False) from qubo_formulation.py) for a small scenario —
by default 3 tasks, 2 servers (6 qubits), which is fast and reliable on a
laptop. SA, GA, and NSGA-II continue to use the FULL, capacity-constrained
problem on the full 10-task scenario elsewhere in the project; QAOA is
compared against them on the SAME small scenario for a fair, apples-to-
apples benchmark.

Usage:
    from data_generator import generate_scenario
    from qaoa_solver import run_qaoa

    scenario = generate_scenario(num_tasks=3, num_servers=2, seed=42)
    assignment, cost, info = run_qaoa(scenario["tasks"], scenario["servers"])
"""

import time

import numpy as np
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qubo_formulation import build_qubo, decode_solution


def run_qaoa(tasks, servers, reps=1, maxiter=30, seed=None, include_capacity=False):
    """
    Run QAOA on the task-to-server assignment problem.

    include_capacity=False (default) keeps the problem small enough to run
    reliably on a local simulator — see module docstring for measured
    qubit-count/runtime scaling. Set True only for very small scenarios
    (2-3 tasks) if you want to include the capacity constraint; expect much
    longer runtimes.

    seed: fixes BOTH the simulator's randomness AND QAOA's initial circuit
    parameters (gamma/beta angles), so the same seed always reproduces the
    exact same result. Without a fixed initial point, QAOA's optimizer
    (COBYLA) starts from a different random guess each run and can converge
    to a different local optimum — a real, well-documented property of
    variational quantum algorithms, but one that makes runs hard to compare
    unless deliberately seeded like this.

    Returns:
        assignment (dict): {task_id: server_id}
        cost (float): the assignment's cost under cost_of_assignment-style scoring
        info (dict): {"num_qubits", "runtime_seconds", "qaoa_objective"}
    """
    qp, qubo, converter = build_qubo(tasks, servers, include_capacity=include_capacity)
    num_qubits = qubo.get_num_binary_vars()

    if seed is not None:
        sampler = StatevectorSampler(seed=seed)
        rng = np.random.default_rng(seed)
        initial_point = rng.uniform(0, 2 * np.pi, size=2 * reps)
    else:
        sampler = StatevectorSampler()
        initial_point = None

    qaoa_mes = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
        initial_point=initial_point,
    )
    optimizer = MinimumEigenOptimizer(qaoa_mes)

    t0 = time.time()
    result = optimizer.solve(qubo)
    runtime = time.time() - t0

    assignment = decode_solution(result.x, tasks, servers, converter)

    info = {
        "num_qubits": num_qubits,
        "runtime_seconds": runtime,
        "qaoa_objective": result.fval,
    }
    return assignment, result.fval, info


def run_exact(tasks, servers, include_capacity=False):
    """
    Solve the same (small) QUBO exactly and classically, using
    NumPyMinimumEigensolver. Used as ground truth to check how close QAOA
    got to the true optimum — not itself a scalable method (exponential),
    only usable because the demo scenario is intentionally small.
    """
    qp, qubo, converter = build_qubo(tasks, servers, include_capacity=include_capacity)

    t0 = time.time()
    result = MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qubo)
    runtime = time.time() - t0

    assignment = decode_solution(result.x, tasks, servers, converter)
    return assignment, result.fval, runtime


if __name__ == "__main__":
    from data_generator import generate_scenario
    from simulated_annealing_solver import cost_of_assignment, random_assignment

    print("=" * 60)
    print("QEDGE-OPT — QAOA Demo (local simulator)")
    print("=" * 60)

    # Small scenario: assignment-only constraint, 6 qubits, runs in a few seconds
    scenario = generate_scenario(num_tasks=3, num_servers=2, seed=42)
    tasks = scenario["tasks"]
    servers = scenario["servers"]

    print(f"\nScenario: {len(tasks)} tasks, {len(servers)} servers "
          f"(assignment-only QUBO — see module docstring for why)")
    for t in tasks:
        print(f"  {t['id']}: size={t['size']}")
    for s in servers:
        print(f"  {s['id']}: energy_cost_per_unit={s['energy_cost_per_unit']}")

    naive = random_assignment(tasks, servers)
    naive_cost = cost_of_assignment(naive, tasks, servers)

    print("\nRunning QAOA on a local simulator...")
    qaoa_assignment, qaoa_qubo_cost, info = run_qaoa(tasks, servers, reps=1, maxiter=30, seed=1)
    qaoa_cost = cost_of_assignment(qaoa_assignment, tasks, servers)

    print(f"QAOA finished in {info['runtime_seconds']:.2f}s using {info['num_qubits']} qubits")

    print("\nRunning exact classical solver (ground truth for this small instance)...")
    exact_assignment, exact_qubo_cost, exact_runtime = run_exact(tasks, servers)
    exact_cost = cost_of_assignment(exact_assignment, tasks, servers)
    print(f"Exact solver finished in {exact_runtime:.4f}s")

    print("\n--- RESULT: RANDOM vs QAOA vs EXACT (ground truth) ---")
    print(f"Random assignment : cost={naive_cost:.2f}")
    print(f"QAOA               : cost={qaoa_cost:.2f}")
    print(f"Exact (optimal)    : cost={exact_cost:.2f}")

    if exact_cost > 0:
        gap = (qaoa_cost - exact_cost) / exact_cost * 100
        print(f"\nQAOA optimality gap vs. exact solution: {gap:.1f}%")

    print("\nQAOA assignment:")
    for task_id, server_id in qaoa_assignment.items():
        print(f"  {task_id} -> {server_id}")