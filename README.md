# QEDGE-OPT: Quantum-Assisted Edge Resource Allocation

**Capstone Project — 7th & 8th Semester**
Domain: Quantum Computing / Edge Computing / Optimization

## Problem

Deciding which computing task should run on which edge server, under
capacity, latency, and energy constraints, is a combinatorial optimization
problem. This project builds and fairly benchmarks four solvers for this
problem: Simulated Annealing, Genetic Algorithm, NSGA-II (all classical),
and QAOA (quantum-assisted, via Qiskit).

## Status: Review 2 — 60% implementation

All four planned solvers are implemented, tested, and compared:
- `data_generator.py` — generates synthetic tasks and edge servers
- `simulated_annealing_solver.py` — classical baseline solver
- `genetic_algorithm_solver.py` — second classical solver (tournament
  selection, single-point crossover, mutation)
- `nsga2_solver.py` — multi-objective classical solver (cost + latency
  violation), using non-dominated sorting and crowding distance
- `qubo_formulation.py` — QUBO formulation of the assignment problem,
  built with Qiskit Optimization (constraints converted to penalty terms
  + binary-encoded slack variables automatically)
- `qaoa_solver.py` — QAOA solver, run on a local Qiskit simulator
- `demo.py` — runnable demo comparing all four solvers, with result charts

### How to run the demo

```bash
pip install matplotlib qiskit qiskit-optimization qiskit-algorithms
python3 demo.py
```

Prints the full SA/GA/NSGA-II comparison on the 10-task scenario, then the
QAOA-vs-exact comparison on a smaller scenario, and saves six charts:
- `annealing_progress.png` — SA cost decreasing over the annealing process
- `ga_progress.png` — GA cost decreasing over generations
- `nsga2_progress.png` — NSGA-II best cost in the Pareto front over generations
- `nsga2_pareto.png` — NSGA-II's final Pareto front (cost vs. latency violation)
- `method_comparison.png` — bar chart comparing SA/GA/NSGA-II cost
- `qaoa_comparison.png` — bar chart comparing QAOA vs. exact solution

### Simulated Annealing (SA) Solver

Classical baseline solver used for Review 1. Optimizes total energy cost of
task-to-server assignment subject to server capacity constraints.

### Genetic Algorithm (GA) Solver

A second solver for the same task-to-server assignment problem, using
tournament selection, single-point crossover, and mutation over a
population of 60 candidates across 200 generations. Shares the exact
same cost function as the Simulated Annealing solver (`cost_of_assignment`),
so results are directly comparable.

### NSGA-II Solver

A multi-objective solver extending GA to optimize **two** objectives at
once: total cost, and latency violation (how much each task's assigned
server's estimated latency, modeled as increasing with server load, exceeds
that task's `latency_requirement`). Uses fast non-dominated sorting and
crowding distance to maintain a diverse Pareto front of trade-off
solutions, plus a "best compromise" pick for direct comparison against
SA/GA's single-number results.

### QUBO Formulation

Expresses the assignment problem as a Quadratic Unconstrained Binary
Optimization model — the format QAOA requires. Uses Qiskit Optimization's
`QuadraticProgramToQubo` to automatically convert the assignment and
capacity constraints into penalty terms (with binary-encoded slack
variables for the capacity inequality), rather than hand-deriving the
expansion. Verified correct: the QUBO's exact classical solution matches
`cost_of_assignment()`'s output exactly on a test scenario.

### QAOA Solver

Runs on a **local Qiskit simulator** (no IBM Quantum account needed).
Important scoping note: the full capacity-constrained QUBO needs 20+
qubits for the real 10-task scenario, which is too large to simulate
quickly. Measured on this project's own hardware:

| Qubits | QAOA runtime (local simulator) |
|---|---|
| 4 | ~0.6s |
| 6 | ~2.8s |
| 12 | did not finish in 60+ seconds |

This is a genuine, reportable characteristic of NISQ-era variational
algorithms, not a bug. QAOA is therefore run on a smaller, assignment-only
scenario (3 tasks, 2 servers, 6 qubits) and compared against an **exact
classical solver** on the same reduced problem for a fair ground-truth
check.

**Result:** QAOA matched the exact optimal solution exactly — **0.0%
optimality gap** — reproducibly, once both the simulator's randomness and
QAOA's initial circuit parameters are seeded (without a fixed initial
point, QAOA's optimizer can converge to different local optima on
different runs — another real, observed property of variational quantum
algorithms worth noting in the report).

**Results on the standard 10-task test scenario (seed=42):**

| Method | Total Cost (lower = better) | Latency Violation (lower = better) | Cost Improvement over Random |
|---|---|---|---|
| Random assignment | 408.53 | 13.34 | — |
| Simulated Annealing | 200.58 | not optimized | 50.9% |
| Genetic Algorithm | 200.15 | not optimized | 51.0% |
| NSGA-II (best compromise) | 200.15 | 11.55 | 51.0% |

**QAOA results on a small 3-task scenario:**

| Method | Cost | Runtime |
|---|---|---|
| Random assignment | 76.43 | — |
| QAOA (local simulator) | 39.06 | ~4-6s |
| Exact (ground truth) | 39.06 | ~0.01s |

## Planned (Review 3)

- Full benchmarking across all four solvers on identical scaled instances
  (objective value, runtime, constraint violations, scalability)
- Streamlit comparison dashboard
- Validation on the Alibaba Cluster Trace v2017 (fully real task-and-server
  data) for final results / paper submission
- IEEE paper submission

## Team

3-member team, Dept. of CSE (Data Science), School of Engineering.

## Tech Stack

Python, Qiskit, Qiskit Optimization, OR-Tools, SimPy, NumPy, Pandas,
Streamlit, DEAP/pymoo.
