# QEDGE-OPT: Quantum-Assisted Edge Resource Allocation

**Capstone Project — 7th & 8th Semester**
Domain: Quantum Computing / Edge Computing / Optimization

## Problem

Deciding which computing task should run on which edge server, under
capacity, latency, and energy constraints, is a combinatorial optimization
problem. This project builds and fairly benchmarks four solvers for this
problem: Simulated Annealing, Genetic Algorithm, NSGA-II (all classical),
and QAOA (quantum-assisted, via Qiskit).

## Status: Post-Review 1 (~30% implementation)

Currently implemented:
- `data_generator.py` — generates synthetic tasks and edge servers
- `simulated_annealing_solver.py` — classical baseline solver
- `genetic_algorithm_solver.py` — second classical solver, using tournament
  selection, single-point crossover, and mutation
- `demo.py` — runnable demo comparing random vs. Simulated-Annealing vs.
  Genetic-Algorithm assignments, with result charts

### How to run the demo

```bash
pip install matplotlib
python3 demo.py
```

This will print the generated tasks/servers, the naive vs. SA vs. GA cost
comparison, the final task-to-server assignments for both solvers, and save
three charts:
- `annealing_progress.png` — SA cost decreasing over the annealing process
- `ga_progress.png` — GA cost decreasing over generations
- `method_comparison.png` — bar chart comparing all three methods

### Simulated Annealing (SA) Solver

Classical baseline solver used for Review 1. Optimizes total energy cost of
task-to-server assignment subject to server capacity constraints.

### Genetic Algorithm (GA) Solver

A second solver for the same task-to-server assignment problem, using
tournament selection, single-point crossover, and mutation over a
population of 60 candidates across 200 generations. It shares the exact
same cost function as the Simulated Annealing solver (`cost_of_assignment`),
so results are directly comparable.

**Results on the standard test scenario (10 tasks, 3 servers, seed=42):**

| Method | Total Cost (lower = better) | Improvement over Random |
|---|---|---|
| Random assignment | 408.53 | — |
| Simulated Annealing | 200.58 | 50.9% |
| Genetic Algorithm | 200.15 | 51.0% |

Both metaheuristics converge to nearly identical costs (within 0.2% of each
other), suggesting they are approaching the true optimum for this scenario.

Run it standalone with:
```bash
python3 genetic_algorithm_solver.py
```

## Planned (Review 2 & 3)

- QUBO formulation of the problem
- NSGA-II solver (multi-objective: cost + latency)
- QAOA solver (Qiskit)
- Full benchmarking across all four solvers (objective value, runtime,
  constraint violations, scalability, noise sensitivity)
- Streamlit comparison dashboard

## Team

3-member team, Dept. of CSE (Data Science), School of Engineering.

## Tech Stack

Python, Qiskit, OR-Tools, SimPy, NumPy, Pandas, Streamlit, DEAP/pymoo.
