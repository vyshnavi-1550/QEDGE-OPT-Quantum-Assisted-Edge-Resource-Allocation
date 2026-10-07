"""
noise_sensitivity.py

QEDGE-OPT — QAOA Noise Sensitivity Analysis

Tests how QAOA's solution quality changes as simulated quantum gate-error
noise increases, averaged over multiple random restarts per noise level
(so the trend reflects noise, not just optimizer luck).

The experiment uses: 3 tasks, 2 servers, assignment-only QUBO, QAOAAnsatz,
AerSimulator with depolarizing noise, and COBYLA with a random initial
point in [0, 2*pi) for every parameter.

NOTE ON AN EARLIER VERSION OF THIS FILE: a prior version added a custom
gamma-range rescaling based on the largest Ising coefficient (restricting
gamma to [0, 2*pi/max_coeff)) plus a grid search. That rescaling was the
bug, not a fix — it's too aggressive for a QUBO with many differently-
scaled terms, and excluded the actual optimal angle often enough that
even the ZERO-noise case gave wildly inconsistent results (16-97% gap
across runs of the identical scenario). Removing it and using the full
[0, 2*pi) range with multiple random restarts (simpler, and what was
already validated in qaoa_solver.py) gives a stable, reproducible result.

The exact ground truth is calculated by exhaustive enumeration using the
same assignment-only objective QAOA was given.
"""

import time

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from scipy.optimize import minimize

from qubo_formulation import build_qubo, decode_solution
from simulated_annealing_solver import cost_of_assignment

SINGLE_QUBIT_GATES = ["rz", "sx", "x", "ry", "rx", "h"]
TWO_QUBIT_GATES = ["cx", "cz", "ecr"]

N_RESTARTS = 4   # independent random-initial-point trials per noise level
MAXITER = 40
SHOTS = 1024


def make_noise_model(p1, p2):
    """Depolarizing noise model. p1 = single-qubit error rate, p2 = two-qubit
    error rate (conventionally ~2x p1 on real hardware)."""
    if p1 <= 0 and p2 <= 0:
        return None
    nm = NoiseModel()
    if p1 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), SINGLE_QUBIT_GATES)
    if p2 > 0:
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), TWO_QUBIT_GATES)
    return nm


def run_noisy_qaoa(tasks, servers, p1, p2, reps=1, maxiter=MAXITER, shots=SHOTS, seed=1):
    """
    One QAOA run at a given noise level, one random restart (seeded).

    Returns:
        assignment (dict): {task_id: server_id}, or {} if no restart's
            sampled outcomes decoded to a complete, valid assignment
        cost (float): cost_of_assignment() of the found solution, or
            float("inf") if none was feasible
        info (dict): runtime, qubit count, and which rank in the
            frequency-sorted outcomes the feasible answer was found at
            (0 = the most frequent outcome was already valid)
    """
    qp, qubo, converter = build_qubo(tasks, servers, include_capacity=False)
    ising, offset = qubo.to_ising()
    num_qubits = ising.num_qubits

    ansatz = QAOAAnsatz(cost_operator=ising, reps=reps)
    ansatz.measure_all()

    backend = AerSimulator(noise_model=make_noise_model(p1, p2))
    tqc = transpile(ansatz, backend=backend, optimization_level=1)
    num_params = len(ansatz.parameters)

    def bitstring_energy(bits):
        x = [int(b) for b in bits[::-1]]  # Qiskit's little-endian bit order
        return qubo.objective.evaluate(x)

    def expected_cost(params):
        bound = tqc.assign_parameters(params)
        result = backend.run(bound, shots=shots, seed_simulator=seed).result()
        counts = result.get_counts()
        total = sum(counts.values())
        return sum(bitstring_energy(b) * c for b, c in counts.items()) / total

    t0 = time.time()
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=num_params)  # full range, no restrictive scaling
    result = minimize(expected_cost, x0, method="COBYLA", options={"maxiter": maxiter})

    bound = tqc.assign_parameters(result.x)
    counts = backend.run(bound, shots=shots, seed_simulator=seed).result().get_counts()
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    runtime = time.time() - t0

    assignment, cost, feasible_rank = None, None, None
    for rank, (bits, _) in enumerate(ranked):
        x_solution = [int(b) for b in bits[::-1]]
        try:
            candidate = decode_solution(np.array(x_solution), tasks, servers, converter)
            if len(candidate) == len(tasks):
                assignment = candidate
                cost = cost_of_assignment(assignment, tasks, servers)
                feasible_rank = rank
                break
        except (KeyError, IndexError):
            continue

    if assignment is None:
        assignment, cost, feasible_rank = {}, float("inf"), None

    info = {
        "runtime_seconds": runtime, "num_qubits": num_qubits,
        "p1": p1, "p2": p2, "feasible_rank": feasible_rank,
    }
    return assignment, cost, info


def run_noise_sweep(tasks, servers, noise_levels=None, seed=1, n_restarts=N_RESTARTS):
    """
    Run QAOA across a range of single-qubit gate error rates, averaging
    over n_restarts independent random initial points at EACH noise level
    (same restart seeds reused across levels, for a paired comparison).
    Returns one summary dict per noise level.
    """
    if noise_levels is None:
        # Realistic hardware error rates (0-5%) plus deliberately exaggerated
        # levels (10-50%) to find where solution quality actually degrades.
        noise_levels = [0.0, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]

    master_rng = np.random.default_rng(seed)
    restart_seeds = master_rng.integers(0, 1_000_000, size=n_restarts).tolist()

    summary = []
    for p1 in noise_levels:
        p2 = p1 * 2
        t_level0 = time.time()
        costs = []

        for restart_seed in restart_seeds:
            _, cost, info = run_noisy_qaoa(tasks, servers, p1, p2, seed=restart_seed)
            if cost != float("inf"):
                costs.append(cost)

        n_feasible = len(costs)
        if n_feasible > 0:
            mean_cost = float(np.mean(costs))
            std_cost = float(np.std(costs))
            min_cost, max_cost = float(min(costs)), float(max(costs))
        else:
            mean_cost = std_cost = min_cost = max_cost = float("inf")

        level_time = time.time() - t_level0
        summary.append({
            "p1": p1, "p2": p2, "mean_cost": mean_cost, "std_cost": std_cost,
            "min_cost": min_cost, "max_cost": max_cost,
            "n_feasible": n_feasible, "n_restarts": n_restarts,
        })

        cost_str = (f"{mean_cost:.2f} (std {std_cost:.2f}, range [{min_cost:.2f}, {max_cost:.2f}])"
                    if n_feasible > 0 else "INFEASIBLE on every restart")
        print(f"  p1={p1:.3f} (p2={p2:.3f}): cost={cost_str}  "
              f"[{n_feasible}/{n_restarts} feasible]  ({level_time:.1f}s)")

    return summary


def plot_noise_sensitivity(summary, exact_cost=None, filepath="noise_sensitivity.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feasible_levels = [r for r in summary if r["n_feasible"] > 0]
    plot_cap = max([r["mean_cost"] for r in feasible_levels], default=1) * 1.5

    p1s = [r["p1"] for r in summary]
    means = [r["mean_cost"] if r["n_feasible"] > 0 else plot_cap for r in summary]
    stds = [r["std_cost"] if r["n_feasible"] > 0 else 0 for r in summary]
    all_infeasible = [r["n_feasible"] == 0 for r in summary]

    plt.figure(figsize=(8, 5))
    plt.errorbar(p1s, means, yerr=stds, marker="o", color="#C0392B", linewidth=2,
                 capsize=4, label="QAOA cost (mean +/- std across restarts)")
    labeled = False
    for p1, m, inf in zip(p1s, means, all_infeasible):
        if inf:
            plt.scatter([p1], [m], marker="x", color="black", s=100, zorder=5,
                        label="No feasible outcome in any restart" if not labeled else None)
            labeled = True

    if exact_cost is not None:
        plt.axhline(exact_cost, color="#2C3E50", linestyle="--", linewidth=1.5,
                    label="Exact optimum (noiseless baseline)")

    plt.xlabel("Single-qubit gate error rate")
    plt.ylabel("Solution cost (lower = better)")
    plt.title(f"QAOA Noise Sensitivity ({summary[0]['n_restarts']} restarts per level)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Chart saved as {filepath}")


if __name__ == "__main__":
    from data_generator import generate_scenario
    from qaoa_solver import run_exact_bruteforce

    print("=" * 60)
    print("QEDGE-OPT — QAOA Noise Sensitivity Analysis")
    print("=" * 60)

    scenario = generate_scenario(num_tasks=3, num_servers=2, seed=42)
    tasks, servers = scenario["tasks"], scenario["servers"]

    print(f"\nScenario: {len(tasks)} tasks, {len(servers)} servers (assignment-only QUBO)")
    print(f"Restarts per noise level: {N_RESTARTS} (COBYLA, maxiter={MAXITER}, shots={SHOTS})")
    print("\nRunning QAOA across increasing simulated gate-error rates...")

    summary = run_noise_sweep(tasks, servers, seed=1, n_restarts=N_RESTARTS)

    _, exact_cost, _ = run_exact_bruteforce(tasks, servers, include_capacity=False)
    print(f"\nExact optimum (noiseless, same assignment-only objective): {exact_cost:.2f}")

    print("\n--- SUMMARY ---")
    for row in summary:
        if row["n_feasible"] == 0:
            print(f"  Gate error {row['p1']:.3f}: INFEASIBLE on every restart")
        else:
            gap = (row["mean_cost"] - exact_cost) / exact_cost * 100 if exact_cost else 0
            gap_std = row["std_cost"] / exact_cost * 100 if exact_cost else 0
            print(f"  Gate error {row['p1']:.3f}: mean gap={gap:+.2f}% (std {gap_std:.2f}%)")

    plot_noise_sensitivity(summary, exact_cost=exact_cost)