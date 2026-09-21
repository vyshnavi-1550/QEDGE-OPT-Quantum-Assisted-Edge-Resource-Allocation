# QEDGE-OPT: Quantum-Assisted Edge Resource Allocation

**Capstone Project — 7th & 8th Semester**
Domain: Quantum Computing / Edge Computing / Optimization

## Problem

Deciding which computing task should run on which edge server, under
capacity, latency, and energy constraints, is a combinatorial optimization
problem. This project builds and fairly benchmarks four solvers for this
problem: Simulated Annealing, Genetic Algorithm, NSGA-II (all classical),
and QAOA (quantum-assisted, via Qiskit).

## Status: Post-Review 1, pre-Review 2 (~45% implementation)

Currently implemented:
- `data_generator.py` — generates synthetic tasks and edge servers
- `simulated_annealing_solver.py` — classical baseline solver
- `genetic_algorithm_solver.py` — second classical solver, using tournament
  selection, single-point crossover, and mutation
- `nsga2_solver.py` — multi-objective classical solver (cost + latency
  violation), using non-dominated sorting and crowding distance
- `demo.py` — runnable demo comparing random vs. SA vs. GA vs. NSGA-II
  assignments, with result charts

### How to run the demo

```bash
pip install matplotlib
python3 demo.py
```

This prints the generated tasks/servers, the naive vs. SA vs. GA vs. NSGA-II
comparison, the final task-to-server assignments, and saves five charts:
- `annealing_progress.png` — SA cost decreasing over the annealing process
- `ga_progress.png` — GA cost decreasing over generations
- `nsga2_progress.png` — NSGA-II best cost in the Pareto front over generations
- `nsga2_pareto.png` — NSGA-II's final Pareto front (cost vs. latency violation)
- `method_comparison.png` — bar chart comparing cost across all methods

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
once:
1. **Total cost** — the same `cost_of_assignment()` used by SA and GA
2. **Latency violation** — how much each task's assigned server's
   estimated latency (modeled as increasing with server load/congestion)
   exceeds that task's `latency_requirement`, summed across all tasks

Uses fast non-dominated sorting and crowding distance to maintain a diverse
Pareto front of trade-off solutions, rather than a single answer. A "best
compromise" pick (lowest cost among zero-violation solutions) is reported
for direct comparison against SA/GA's single-number results.

**Results on the standard test scenario (10 tasks, 3 servers, seed=42):**

| Method | Total Cost (lower = better) | Latency Violation (lower = better) | Cost Improvement over Random |
|---|---|---|---|
| Random assignment | 408.53 | 13.34 | — |
| Simulated Annealing | 200.58 | not optimized | 50.9% |
| Genetic Algorithm | 200.15 | not optimized | 51.0% |
| NSGA-II (best compromise) | 200.15 | 11.55 | 51.0% |

NSGA-II's best-compromise solution matches GA's cost exactly, while also
reducing latency violation by ~13% versus random — a genuine multi-objective
improvement that single-objective SA and GA don't account for. The final
Pareto front contains dozens of trade-off solutions ranging from low-cost/
higher-violation to higher-cost/lower-violation, giving a full picture of
the cost-latency trade-off space rather than one fixed answer.

Run it standalone with:
```bash
python3 nsga2_solver.py
```

## Planned (Review 2 & 3)

- QUBO formulation of the problem
- QAOA solver (Qiskit)
- Full benchmarking across all four solvers (objective value, runtime,
  constraint violations, scalability, noise sensitivity)
- Streamlit comparison dashboard
- Validation on the Alibaba Cluster Trace v2017 (fully real task-and-server
  data) — planned for Review 3 / final results

## Team

3-member team, Dept. of CSE (Data Science), School of Engineering.

## Tech Stack

Python, Qiskit, OR-Tools, SimPy, NumPy, Pandas, Streamlit, DEAP/pymoo.
