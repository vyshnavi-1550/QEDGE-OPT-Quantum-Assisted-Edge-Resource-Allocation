# QEDGE-OPT: Quantum-Assisted Edge Resource Allocation

**Capstone Project, 7th & 8th Semester**
**Domain:** Quantum Computing / Edge Computing / Optimization

## Problem

Deciding which computing task should run on which edge server, under capacity,
latency, and energy constraints, is a combinatorial optimization problem. This
project builds and fairly benchmarks five approaches to it: Simulated Annealing,
Genetic Algorithm, NSGA-II (all classical), QAOA (quantum-assisted, via Qiskit),
and an exact MILP solver (HiGHS) used as a ground-truth optimality baseline.

## Status: Review 2, about 85% implementation

Implemented, tested, and benchmarked:

- `data_generator.py`: generates synthetic tasks and edge servers
- Alibaba Cluster Trace loader: real task and server data, with a capacity floor
- `simulated_annealing_solver.py`: classical baseline solver
- `genetic_algorithm_solver.py`: tournament selection, single-point crossover, mutation
- `nsga2_solver.py`: multi-objective solver (cost + latency violation), using
  non-dominated sorting and crowding distance
- `qubo_formulation.py`: QUBO formulation built with Qiskit Optimization
  (constraints converted to penalty terms with binary-encoded slack variables)
- `qaoa_solver.py`: QAOA on a local Qiskit simulator, with hybrid repair and a control experiment
- Exact baseline: HiGHS MILP (OR-Tools optional) for optimality-gap measurement
- `run_benchmark.py`: multi-size benchmark on both datasets (cost, violations,
  runtime, scalability, optimality gap)
- `demo.py`: runnable demo comparing the solvers, with result charts

## How to run

```bash
pip install matplotlib qiskit qiskit-optimization qiskit-algorithms
python3 demo.py              # demo + charts
python3 run_benchmark.py     # full multi-size benchmark, writes CSV results
```

`demo.py` prints the SA/GA/NSGA-II comparison on the 10-task scenario, then the
QAOA-vs-exact comparison on a smaller scenario, and saves six charts:

- `annealing_progress.png`: SA cost over the annealing process
- `ga_progress.png`: GA cost over generations
- `nsga2_progress.png`: NSGA-II best cost in the Pareto front over generations
- `nsga2_pareto.png`: NSGA-II's final Pareto front (cost vs. latency violation)
- `method_comparison.png`: bar chart comparing SA/GA/NSGA-II cost
- `qaoa_comparison.png`: bar chart comparing QAOA vs. exact solution

## Solvers

### Simulated Annealing (SA)
Classical baseline. Optimizes total energy cost of task-to-server assignment
subject to server capacity constraints.

### Genetic Algorithm (GA)
Tournament selection, single-point crossover, and mutation over a population of
60 candidates across 200 generations. Shares the same cost function as SA
(`cost_of_assignment`), so results are directly comparable.

### NSGA-II
Multi-objective extension of GA optimizing total cost and latency violation
(how much each task's assigned server's estimated latency, modeled as increasing
with server load, exceeds that task's `latency_requirement`). Maintains a
diverse Pareto front, plus a "best compromise" pick for comparison with SA/GA.

### Exact baseline (HiGHS MILP)
Finds the true optimum of the cost objective. Used to compute each heuristic's
optimality gap. It ignores latency, so NSGA-II's gap partly reflects cost it
deliberately trades for lower latency.

### QUBO formulation
Expresses the assignment problem as a QUBO, the format QAOA requires, using
Qiskit Optimization's `QuadraticProgramToQubo`. Verified: the QUBO's exact
classical solution matches `cost_of_assignment()` exactly on a test scenario.

### QAOA
Runs on a local Qiskit simulator (no IBM Quantum account needed). The full
capacity-constrained QUBO needs 20+ qubits for the 10-task scenario, which is
too large to simulate quickly. Measured runtimes:

| Qubits | QAOA runtime (local simulator) |
|---|---|
| 4 | ~0.6 s |
| 6 | ~2.8 s |
| 12 | did not finish in 60+ s |

This is a real characteristic of NISQ-era variational algorithms, not a bug.
QAOA is therefore run on small, assignment-only instances.

- **On its own problem:** on the 3-task, 2-server scenario QAOA matched the exact
  optimum (0.0% gap), reproducibly once the simulator and QAOA's initial
  parameters are seeded. Without a fixed initial point the optimizer can
  converge to different local optima, another observed property of variational
  algorithms.
- **Against the full objective:** because the QUBO encodes assignment only, QAOA
  can overload servers. Scored with the capacity penalty, the gap is large and
  is driven by capacity overflow, not by QAOA failing. The benchmark reports
  both `gap_on_own_problem_pct` and `gap_real_world_pct`.

## Results

### 10-task scenario (seed=42)

| Method | Total Cost (lower = better) | Latency Violation | Improvement over Random |
|---|---|---|---|
| Random assignment | 408.53 | 13.34 | n/a |
| Simulated Annealing | 200.58 | not optimized | 50.9% |
| Genetic Algorithm | 200.15 | not optimized | 51.0% |
| NSGA-II (best compromise) | 200.15 | 11.55 | 51.0% |

### Optimality gap vs. exact solver (mean over 3 seeds, Alibaba / synthetic)

| Tasks | SA | GA | NSGA-II |
|---|---|---|---|
| 10 | 0.41% / 0.07% | 1.26% / 0.07% | 0.39% / 0.74% |
| 25 | 0.35% / 0.27% | 0.30% / 0.53% | 0.19% / 0.87% |
| 50 | 0.15% / 0.14% | 1.41% / 1.45% | 2.49% / 1.18% |
| 100 | 0.26% / 0.04% | 3.84% / 2.65% | 3.84% / 2.62% |

Takeaways:

- SA is within about 0.4% of optimal at every size.
- GA and NSGA-II drift to about 2.6-3.8% at 100 tasks.
- At about 100 tasks the exact solver finds the optimum in roughly 1.4-1.8 s, so
  it is a strong baseline at these sizes. Metaheuristics earn their place mainly
  for the multi-objective version and for larger or non-linear extensions.

### QAOA on a small 3-task scenario

| Method | Cost | Runtime |
|---|---|---|
| Random assignment | 76.43 | n/a |
| QAOA (local simulator) | 39.06 | ~4-6 s |
| Exact (ground truth) | 39.06 | ~0.01 s |

## Remaining work (before Review 3)

- Noise sensitivity analysis for QAOA (Qiskit noise model)
- Streamlit comparison dashboard
- IEEE-style conference paper

## Team

3-member team, Dept. of CSE (Data Science), School of Engineering.

## Tech Stack

Python, Qiskit, Qiskit Optimization, OR-Tools, HiGHS, NumPy, Pandas, Streamlit.
