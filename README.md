# QEDGE-OPT: Quantum-Assisted Edge Resource Allocation

**Capstone Project — 7th & 8th Semester**
Domain: Quantum Computing / Edge Computing / Optimization

## Problem

Deciding which computing task should run on which edge server, under
capacity, latency, and energy constraints, is a combinatorial optimization
problem. This project builds and fairly benchmarks four solvers for this
problem: Simulated Annealing, Genetic Algorithm, NSGA-II (all classical),
and QAOA (quantum-assisted, via Qiskit).

## Status: Review 1 (20% implementation)

Currently implemented:
- `data_generator.py` — generates synthetic tasks and edge servers
- `simulated_annealing_solver.py` — classical baseline solver
- `demo.py` — runnable demo comparing a random assignment vs. the
  Simulated-Annealing-optimized assignment, with a results chart

### How to run the demo

```bash
pip install matplotlib
python3 demo.py
```

This will print the generated tasks/servers, the naive vs. optimized cost
comparison, the final task-to-server assignment, and save a chart
(`annealing_progress.png`) showing the cost decreasing over the annealing
process.

## Planned (Review 2 & 3)

- QUBO formulation of the problem
- Genetic Algorithm and NSGA-II solvers
- QAOA solver (Qiskit)
- Full benchmarking across all four solvers (objective value, runtime,
  constraint violations, scalability, noise sensitivity)
- Streamlit comparison dashboard

## Team

3-member team, Dept. of CSE (Data Science), School of Engineering.

## Tech Stack

Python, Qiskit, OR-Tools, SimPy, NumPy, Pandas, Streamlit, DEAP/pymoo.
