"""
exact_baseline.py
------------------
Exact optimality reference for the classical solvers.

For every (size, seed) instance in benchmark_<data>.csv it rebuilds the same
scenario, solves the capacity-constrained assignment problem to optimality
(minimum energy cost, capacity as a HARD constraint), and reports how far
SA, GA and NSGA-II are from that optimum.

Backends:
    scipy    (default) SciPy's HiGHS MILP solver - no extra install
    ortools  OR-Tools CP-SAT  (pip install ortools)

Running both backends must give the same optimum - a useful cross-check.

Safety checks (printed if triggered):
    - the exact objective is compared with cost_of_assignment() on the same
      assignment; a mismatch means the cost model has terms this baseline
      does not model
    - a solver beating the "optimum" (negative gap) also signals a mismatch

Usage:
    python exact_baseline.py                    # Alibaba benchmark CSV
    python exact_baseline.py --data synthetic
    python exact_baseline.py --backend ortools
"""

import argparse
import csv
import time
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_benchmark import get_scenario
from simulated_annealing_solver import cost_of_assignment

METHODS = ["SA", "GA", "NSGA-II"]
COLORS = {"SA": "#1F4E78", "GA": "#8E44AD", "NSGA-II": "#C0392B"}


def solve_scipy(tasks, servers, time_limit):
    from scipy.optimize import milp, LinearConstraint, Bounds

    n, m = len(tasks), len(servers)
    sizes = np.array([t["size"] for t in tasks], dtype=float)
    energy = np.array([s["energy_cost_per_unit"] for s in servers], dtype=float)
    caps = np.array([s["capacity"] for s in servers], dtype=float)

    c = np.outer(sizes, energy).ravel()          # variable index = i*m + j

    a_assign = np.zeros((n, n * m))
    for i in range(n):
        a_assign[i, i * m:(i + 1) * m] = 1.0
    a_cap = np.zeros((m, n * m))
    for j in range(m):
        for i in range(n):
            a_cap[j, i * m + j] = sizes[i]

    res = milp(
        c,
        constraints=[LinearConstraint(a_assign, 1, 1),
                     LinearConstraint(a_cap, -np.inf, caps)],
        integrality=np.ones(n * m),
        bounds=Bounds(0, 1),
        options={"time_limit": time_limit},
    )
    if res.x is None:
        return None, None, "infeasible"

    x = np.round(res.x).reshape(n, m)
    assign = {tasks[i]["id"]: servers[int(x[i].argmax())]["id"] for i in range(n)}
    status = "optimal" if res.status == 0 else f"limit(status={res.status})"
    return assign, float(res.fun), status


def solve_cpsat(tasks, servers, time_limit):
    from ortools.sat.python import cp_model

    n, m = len(tasks), len(servers)
    model = cp_model.CpModel()
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(n) for j in range(m)}

    for i in range(n):
        model.AddExactlyOne(x[i, j] for j in range(m))
    for j, s in enumerate(servers):
        model.Add(sum(int(round(t["size"])) * x[i, j]
                      for i, t in enumerate(tasks)) <= int(round(s["capacity"])))

    # energy costs have two decimals -> scale by 100 for exact integers
    model.Minimize(sum(
        int(round(t["size"] * servers[j]["energy_cost_per_unit"] * 100)) * x[i, j]
        for i, t in enumerate(tasks) for j in range(m)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, "infeasible"

    assign = {}
    for i, t in enumerate(tasks):
        for j, s in enumerate(servers):
            if solver.Value(x[i, j]):
                assign[t["id"]] = s["id"]
    name = "optimal" if status == cp_model.OPTIMAL else "feasible(limit)"
    return assign, solver.ObjectiveValue() / 100.0, name


def read_benchmark(path):
    groups = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["num_tasks"]), int(r["num_servers"]), int(r["seed"]))
            groups[key][r["method"]] = float(r["cost"])
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["alibaba", "synthetic"], default="alibaba")
    parser.add_argument("--backend", choices=["scipy", "ortools"], default="scipy")
    parser.add_argument("--time-limit", type=float, default=60.0)
    args = parser.parse_args()

    solve = solve_scipy if args.backend == "scipy" else solve_cpsat
    bench_path = f"benchmark_{args.data}.csv"
    groups = read_benchmark(bench_path)
    print(f"Read {bench_path}: {len(groups)} instances, backend={args.backend}")

    rows = []
    mismatches = negatives = 0

    for (n, m, seed), costs in sorted(groups.items()):
        scenario = get_scenario(args.data, n, m, seed)
        tasks, servers = scenario["tasks"], scenario["servers"]

        start = time.perf_counter()
        assign, model_obj, status = solve(tasks, servers, args.time_limit)
        runtime = time.perf_counter() - start

        if assign is None:
            print(f"  {n}T/{m}S seed {seed}: no feasible assignment, skipped")
            continue

        ref = cost_of_assignment(assign, tasks, servers)
        if abs(ref - model_obj) > 1e-6 * max(1.0, abs(ref)):
            mismatches += 1
        print(f"  {n:3d}T/{m}S seed {seed}: optimum={ref:9.2f}  "
              f"{status}  ({runtime:.2f}s)")

        for method in METHODS:
            if method not in costs:
                continue
            gap = (costs[method] - ref) / ref * 100 if ref else 0.0
            if gap < -0.01:
                negatives += 1
            rows.append({
                "data": args.data, "num_tasks": n, "num_servers": m, "seed": seed,
                "method": method, "cost": round(costs[method], 2),
                "exact_cost": round(ref, 2), "gap_pct": round(gap, 2),
                "exact_runtime_s": round(runtime, 3), "exact_status": status,
                "backend": args.backend,
            })

    if not rows:
        print("No results.")
        return

    out_csv = f"exact_gap_{args.data}_{args.backend}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {out_csv}")

    bucket = defaultdict(list)
    rt = defaultdict(list)
    for r in rows:
        bucket[(r["num_tasks"], r["method"])].append(r["gap_pct"])
        rt[r["num_tasks"]].append(r["exact_runtime_s"])
    sizes = sorted({r["num_tasks"] for r in rows})

    print("\n" + "=" * 66)
    print(f"GAP TO EXACT OPTIMUM ({args.data}, mean / max over seeds)")
    print("=" * 66)
    print(f"{'Tasks':>5} | {'Method':8} | {'Mean gap %':>10} | {'Max gap %':>9} | {'Exact time(s)':>13}")
    print("-" * 66)
    for n in sizes:
        for method in METHODS:
            g = bucket.get((n, method))
            if g:
                print(f"{n:5d} | {method:8} | {mean(g):10.2f} | {max(g):9.2f} | {mean(rt[n]):13.2f}")
        print("-" * 66)

    if mismatches:
        print(f"WARNING: {mismatches} instance(s) where the exact objective differs "
              f"from cost_of_assignment().\n         The cost model has terms this "
              f"baseline does not model - check before quoting gaps.")
    if negatives:
        print(f"WARNING: {negatives} solver result(s) beat the 'optimum' (negative gap) - "
              f"same cause as above.")

    plt.figure(figsize=(9, 5))
    width = 0.25
    for j, method in enumerate(METHODS):
        xs = [i + (j - 1) * width for i in range(len(sizes))]
        ys = [mean(bucket[(n, method)]) if bucket.get((n, method)) else 0 for n in sizes]
        plt.bar(xs, ys, width, label=method, color=COLORS[method])
    plt.xticks(range(len(sizes)), [str(n) for n in sizes])
    plt.xlabel("Number of tasks")
    plt.ylabel("Mean gap to exact optimum (%)")
    plt.title(f"Optimality gap of classical solvers ({args.data})")
    plt.legend()
    plt.tight_layout()
    out_png = f"exact_gap_{args.data}_{args.backend}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
