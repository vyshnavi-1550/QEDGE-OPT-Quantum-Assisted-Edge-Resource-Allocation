# QEDGE-OPT: Quantum-Assisted Edge Resource Allocation

**Capstone Project, 7th & 8th Semester**
**Domain:** Quantum Computing / Edge Computing / Optimization

## Problem

Deciding which computing task should run on which edge server, under capacity,
latency, and energy constraints, is a combinatorial optimization problem. This
project builds and fairly benchmarks five approaches to it: Simulated Annealing,
Genetic Algorithm, NSGA-II (all classical), QAOA (quantum-assisted, via Qiskit),
and an exact MILP solver (HiGHS via SciPy) used as a ground-truth optimality
baseline.

## Status: Review 2, about 95% implementation

Implemented, tested, and benchmarked:

- `data_generator.py`: generates synthetic tasks and edge servers
- `alibaba_data_loader.py`: loads Alibaba Cluster Trace tasks and servers (with a capacity floor)
- `real_data_loader.py`: earlier real-data loader
- `simulated_annealing_solver.py`: classical baseline solver
- `genetic_algorithm_solver.py`: tournament selection, single-point crossover, mutation
- `nsga2_solver.py`: multi-objective solver (cost + latency violation), using
  non-dominated sorting and crowding distance
- `qubo_formulation.py`: QUBO formulation built with Qiskit Optimization
  (constraints converted to penalty terms with binary-encoded slack variables)
- `qaoa_solver.py`: QAOA on a local Qiskit simulator, with hybrid repair
- `qaoa_control.py`: control experiment comparing QAOA with Exact-QUBO,
  cheapest-server and random baselines
- `exact_baseline.py`: exact MILP baseline (SciPy/HiGHS) for optimality gaps
- `run_benchmark.py`: multi-size benchmark (cost, violations, runtime, scalability)
- `app.py`: Streamlit dashboard that reads the result CSVs and charts
- `demo.py`: runnable demo comparing the solvers, with result charts

## How to run

```bash
pip install matplotlib qiskit qiskit-optimization qiskit-algorithms scipy streamlit pandas
python3 demo.py                  # demo + charts
python3 run_benchmark.py         # multi-size benchmark, writes CSV results and charts
python3 exact_baseline.py        # exact MILP baseline, writes exact_gap_*_scipy.csv
python3 -m streamlit run app.py  # comparison dashboard (opens in browser)
```

On Windows, if `pip` or `streamlit` is not recognized, use
`python -m pip install ...` and `python -m streamlit run app.py`.

`demo.py` prints the SA/GA/NSGA-II comparison on the 10-task scenario, then the
QAOA-vs-exact comparison on a smaller scenario, and saves six charts:

- `annealing_progress.png`: SA cost over the annealing process
- `ga_progress.png`: GA cost over generations
- `nsga2_progress.png`: NSGA-II best cost in the Pareto front over generations
- `nsga2_pareto.png`: NSGA-II's final Pareto front (cost vs. latency violation)
- `method_comparison.png`: bar chart comparing SA/GA/NSGA-II cost
- `qaoa_comparison.png`: bar chart comparing QAOA vs. exact solution

## Dashboard

`app.py` is a Streamlit app with six tabs: Overview, Optimality gap, QAOA, QAOA
control, Charts, and Raw data. It reads the CSV and PNG files produced by the
scripts above (it does not re-run the solvers), so run the benchmark scripts
first if a tab reports a missing file.

## Result files

| File | Contents |
|---|---|
| `benchmark_alibaba.csv`, `benchmark_synthetic.csv` | Per-dataset benchmark of Random/SA/GA/NSGA-II |
| `benchmark_results.csv` | Latest `run_benchmark.py` run (Alibaba) |
| `exact_gap_alibaba_scipy.csv`, `exact_gap_synthetic_scipy.csv` | Optimality gap of each heuristic vs. the exact optimum, per seed |
| `qaoa_alibaba.csv`, `qaoa_synthetic.csv` | QAOA results, overflow, and repaired gap |
| `qaoa_control_alibaba.csv`, `qaoa_control_synthetic.csv` | QAOA control experiment (5 seeds per size) |
| `qaoa_results.csv` | Latest `run_benchmark.py` QAOA run |

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
Finds the true optimum of the cost objective and is used to compute each
heuristic's optimality gap. It ignores latency, so NSGA-II's gap partly
reflects cost it deliberately trades for lower latency.

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
QAOA is therefore run on small, assignment-only instances (2-4 tasks, 2 servers).

- **On its own problem:** on the Alibaba scenarios QAOA reached the QUBO optimum
  for 2 and 4 tasks and fell 28.3% short on the 3-task instance. A control
  experiment over 5 seeds per size (15 runs) shows QAOA reaching the true
  optimum after repair in 12 of 15 runs, versus 11 of 15 for both the exact
  QUBO solution and a cheapest-server baseline. QAOA therefore matches what the
  assignment-only formulation allows but does not outperform a trivial
  heuristic. Without a fixed initial point the optimizer can also converge to
  different local optima on different runs.
- **Against the full objective:** because the QUBO encodes assignment only, QAOA
  overloads servers (overflow of 5, 10 and 18 for 2, 3 and 4 tasks). Hybrid
  repair removes the overflow and brings the gap to 0.0%, 14.0% and 0.0%
  (`qaoa_alibaba.csv`). Unrepaired real-world gaps are very large (898-1531%
  at 3-4 tasks) and are caused by capacity overflow, not by QAOA failing.

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

Source: `exact_gap_alibaba_scipy.csv` and `exact_gap_synthetic_scipy.csv`.

Takeaways:

- SA is within about 0.4% of optimal at every size.
- GA and NSGA-II drift to about 2.6-3.8% at 100 tasks.
- At 100 tasks the exact solver finds the optimum in about 1.4-1.8 s on Alibaba
  data (about 0.05-1.0 s on synthetic), so it is a strong baseline at these
  sizes. Metaheuristics earn their place mainly for the multi-objective version
  and for larger or non-linear extensions.

### QAOA on Alibaba scenarios (`qaoa_alibaba.csv`)

| Tasks / Servers | Qubits | Own-problem gap | Overflow | Repaired gap | Runtime |
|---|---|---|---|---|---|
| 2 / 2 | 4 | 0.0% | 5 | 0.0% | ~2.1 s |
| 3 / 2 | 6 | 28.3% | 10 | 14.02% | ~5.3 s |
| 4 / 2 | 8 | 0.0% | 18 | 0.0% | ~24.7 s |

The exact solver takes under 0.01 s on each of these instances.

## Remaining work (before Review 3)

- Noise sensitivity analysis for QAOA (Qiskit noise model)
- IEEE-style conference paper

## Team

3-member team, Dept. of CSE (Data Science), School of Engineering.

## Tech Stack

Python, Qiskit, Qiskit Optimization, SciPy (HiGHS), OR-Tools (optional), NumPy,
Pandas, Matplotlib, Streamlit.