QEDGE-OPT: Quantum-Assisted Edge Resource Allocation
Capstone Project — 7th & 8th Semester
Domain: Quantum Computing, Edge Computing, and Optimization
1. Problem Statement
Deciding which computing task should run on which edge server under capacity, latency, and energy constraints is a combinatorial optimization problem. QEDGE-OPT develops and compares classical optimization algorithms and a quantum-assisted approach for task-to-server allocation.

The project includes:
- Simulated Annealing (SA)
- Genetic Algorithm (GA)
- NSGA-II
- QAOA (Quantum Approximate Optimization Algorithm) using Qiskit
- Exact baselines for small-instance optimality comparisons
2. Current Implementation Status
The following components are present in the project:
File	Purpose
data_generator.py	Generates synthetic tasks and edge servers
alibaba_data_loader.py	Loads Alibaba Cluster Trace-derived task/server scenarios
real_data_loader.py	Earlier real-data loading utility
simulated_annealing_solver.py	SA implementation and shared assignment-cost function
genetic_algorithm_solver.py	GA with selection, crossover, and mutation
nsga2_solver.py	Multi-objective optimization of cost and latency violation
qubo_formulation.py	Builds the assignment model and converts it to QUBO
qaoa_solver.py	QAOA and exact-QUBO routines
qaoa_control.py	Control experiment for QAOA, Exact-QUBO, cheapest-server, and random baselines
exact_baseline.py	Exact MILP baseline using SciPy/HiGHS
run_benchmark.py	Multi-size benchmark and QAOA benchmark
noise_sensitivity.py	QAOA simulator noise-sensitivity experiment
app.py	Streamlit comparison dashboard
app_advanced.py	Advanced Streamlit dashboard
demo.py	Demonstrates the optimization methods and produces charts


The scripts and output files have been run and checked in the development environment. This status does not mean every feature has undergone a complete independent correctness audit.
3. How to Run
Run these commands from the project directory in an environment with the dependencies installed.
python -m pip install matplotlib qiskit qiskit-optimization qiskit-algorithms scipy streamlit pandas numpy
python demo.py
python run_benchmark.py
python exact_baseline.py
python noise_sensitivity.py
python -m streamlit run app.py
To run the advanced dashboard instead:
python -m streamlit run app_advanced.py
On Windows, use python -m pip and python -m streamlit if pip or streamlit is not recognized. If a script reports a missing dependency, install that package in the Python environment used to run the project.

run_benchmark.py regenerates the latest benchmark_results.csv, benchmark_cost.png, benchmark_runtime.png, qaoa_results.csv, and qaoa_scaling.png. Other scripts generate their own output files.
4. Dashboard
The Streamlit dashboard reads existing CSV and PNG files; it does not automatically rerun every solver. Run the relevant experiment scripts first if a dashboard section reports a missing or outdated output file.
5. Methods
5.1 Simulated Annealing (SA)
SA is a classical stochastic search baseline. The project evaluates assignments using energy-related cost plus a penalty for server-capacity overload. The current cost function uses a capacity-overload penalty weight of 50.
5.2 Genetic Algorithm (GA)
GA searches over a population of candidate task-to-server assignments using evolutionary operators, including selection, crossover, and mutation. It uses the shared assignment-cost function for comparison with SA.
5.3 NSGA-II
NSGA-II optimizes two objectives: assignment cost and latency violation. It uses non-dominated sorting and crowding distance to maintain a diverse Pareto front. The benchmark selects a compromise solution for comparison with single-objective methods.
5.4 Exact baselines
The project includes a SciPy/HiGHS MILP baseline and exact routines for small QUBO or exhaustive-search instances. Be careful to distinguish the exact solver used for the full cost objective from the exact solver used for the assignment-only QUBO objective. They are not interchangeable reference values.
5.5 QUBO formulation
The assignment model uses binary task-to-server decision variables. The QUBO conversion adds penalty terms for the constraints included in the model. Capacity constraints can require binary-encoded slack variables, increasing the number of variables/qubits.
5.6 QAOA and classical capacity repair
Important limitation: QAOA runs with include_capacity=False by default, so the QAOA circuit solves an assignment-only QUBO for small instances. The decoded assignment is then evaluated against the full objective, and a classical greedy capacity-repair step attempts to eliminate capacity overflow.

Therefore, the current method is a hybrid QAOA + classical capacity-repair pipeline. The current experiments do not demonstrate that QAOA directly solves the full capacity-constrained problem or that it provides a quantum runtime advantage.

The benchmark reports three distinct gaps:
- Own-problem gap: QAOA's QUBO objective versus the exact solution of the same QUBO.
- Full-objective gap: the un-repaired assignment scored with the full capacity-penalized objective versus the full-problem optimum.
- Repaired gap: the repaired assignment scored with the full objective versus the full-problem optimum.
6. Latest Classical Benchmark Results
The latest recorded python run_benchmark.py run evaluated Alibaba-derived scenarios using three seeds per size. The values below are means over those seeds.
Tasks	Method	Mean cost	Mean latency violation	Mean capacity overflow	Mean runtime (s)
10	Random	951.33	9.72	15.33	0.00
10	SA	138.42	9.89	0.00	0.16
10	GA	139.17	9.73	0.00	0.22
10	NSGA-II	138.17	9.86	0.00	9.92
25	Random	395.67	17.94	1.00	0.00
25	SA	279.83	22.16	0.00	0.29
25	GA	279.83	22.05	0.00	0.34
25	NSGA-II	279.50	21.39	0.00	11.12
50	Random	1846.04	33.52	21.00	0.00
50	SA	612.91	41.18	0.00	0.37
50	GA	620.55	40.72	0.00	0.41
50	NSGA-II	627.29	38.27	0.00	8.96
100	Random	1879.47	60.59	6.67	0.00
100	SA	1144.23	86.04	0.00	0.66
100	GA	1184.73	75.14	0.00	0.68
100	NSGA-II	1186.13	77.14	0.00	9.95


Observations
- SA, GA, and NSGA-II reported zero mean capacity overflow at all four tested sizes.
- NSGA-II had the lowest mean cost at 10 and 25 tasks in this run.
- SA had the lowest mean cost at 50 and 100 tasks in this run.
- NSGA-II runtime was much higher than SA and GA for these benchmark sizes.
- These results apply to the tested scenarios and settings; they should not be generalized to all workloads.
The latency violation values are the metric reported by the implementation. Use the project's documented latency model and units when interpreting or presenting them.
7. Latest QAOA Benchmark Results
The latest qaoa_results.csv contains three Alibaba-derived instances with two servers.
Tasks / Servers	Qubits	Own-problem gap	Un-repaired capacity overflow	Full-objective gap	Repaired overflow	Repaired gap	QAOA runtime (s)
2 / 2	4	0.00%	5	970.00%	0	0.00%	1.516
3 / 2	6	28.30%	10	898.13%	0	14.02%	4.231
4 / 2	8	0.00%	18	1531.58%	0	0.00%	23.431


Interpretation
- Capacity repair removed overflow in all three recorded QAOA benchmark cases.
- The repaired 3-task case remained 14.02% above the full-problem optimum.
- The own-problem gap is measured against the assignment-only QUBO, whereas the full-objective and repaired gaps use the full cost function. The numbers should not be compared as if they use the same objective.
- QAOA runtimes were longer than the exact QUBO solver runtimes in these small cases. These experiments do not establish a quantum runtime advantage.
8. Control Experiment
The combined qaoa_control_alibaba.csv and qaoa_control_synthetic.csv files contain 30 rows per method. Their recorded hit_optimum aggregates are:
Method	Sum of hit_optimum	Mean
QAOA	23.00	0.7667
Exact-QUBO	23.00	0.7667
Cheapest-server	23.00	0.7667
Random (mean)	14.05	0.4683


The random method stores averaged values, so its sum of 14.05 is not an integer count of successful scenarios. Describe this as an aggregate of the recorded metric, not as 14.05 discrete successful runs. These combined CSV results supersede older control summaries when reporting the latest saved files.
9. Noise Sensitivity
noise_sensitivity.py runs a simulator-based QAOA noise-sensitivity experiment and generates noise_sensitivity.png. These are simulated gate-error experiments, not measurements from real quantum hardware. Results depend on the tested scenario, error settings, optimizer configuration, shots, and restarts.
10. Result Files
File	Contents
benchmark_results.csv	Latest multi-size Alibaba-derived benchmark
benchmark_cost.png	Latest benchmark cost chart
benchmark_runtime.png	Latest benchmark runtime chart
benchmark_alibaba.csv, benchmark_synthetic.csv	Earlier per-dataset benchmark results
exact_gap_alibaba_scipy.csv, exact_gap_synthetic_scipy.csv	Earlier exact-baseline gap results
qaoa_results.csv	Latest small-instance QAOA benchmark
qaoa_control_alibaba.csv, qaoa_control_synthetic.csv	Control-experiment outputs
noise_sensitivity.png	QAOA simulator noise-sensitivity chart
qaoa_scaling.png	Latest QAOA scaling chart
annealing_progress.png	SA progress chart
ga_progress.png	GA progress chart
nsga2_progress.png	NSGA-II progress chart
nsga2_pareto.png	NSGA-II Pareto-front chart
method_comparison.png	Classical method comparison chart
qaoa_comparison.png	QAOA/exact comparison chart


Some files listed above are from earlier runs. Check file timestamps and regenerate outputs before using them in a final paper.
11. Limitations
- QAOA currently omits capacity constraints from its QUBO by default and relies on classical repair.
- QAOA is benchmarked on 2–4 tasks and 2 servers, while classical methods are tested on larger instances.
- The repaired QAOA solution is not always the full-problem optimum.
- The current small-instance simulator runs are slower than the exact QUBO solver.
- Simulator noise experiments do not demonstrate performance on real quantum hardware.
- Alibaba trace preprocessing and the assumptions used to construct tasks and servers should be documented for reproducibility.
- Runtime and solution quality depend on machine hardware, library versions, random seeds, optimizer settings, and generated scenarios.
12. Future Work
- Investigate capacity-aware QUBO formulations with manageable slack-variable overhead.
- Improve and evaluate the classical repair strategy on more small instances.
- Run additional seeds and report variability for QAOA and classical methods.
- Validate dashboard metrics against the latest CSV outputs.
- Document data preprocessing, dependency versions, experiment configurations, and reproduction steps.
- Explore larger instances only where computational resources allow.
13. Technology Stack
Python, Qiskit, Qiskit Optimization, Qiskit Algorithms, SciPy/HiGHS, NumPy, Pandas, Matplotlib, and Streamlit.
14. Repository
QEDGE-OPT: Quantum-Assisted Edge Resource Allocation