"""
run_benchmark.py
-----------------
Benchmarks SA, GA and NSGA-II (plus a random baseline) across several
problem sizes and seeds, and QAOA vs exact on small instances.

Outputs:
    benchmark_results.csv   one row per (size, seed, method)
    qaoa_results.csv        one row per small QAOA instance
    benchmark_cost.png      mean cost vs problem size
    benchmark_runtime.png   mean runtime vs problem size (log scale)
    qaoa_scaling.png        QAOA vs exact runtime vs qubits (log scale)

Usage:
    python run_benchmark.py                   # Alibaba data (default)
    python run_benchmark.py --data synthetic  # synthetic generator
    python run_benchmark.py --skip-qaoa       # classical solvers only
"""

import argparse
import csv
import itertools
import math
import random
import time
import warnings
from collections import defaultdict
from statistics import mean

# Qiskit's scipy sparse-matrix warnings are harmless noise
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alibaba_data_loader import load_alibaba_scenario, CAPACITY_HEADROOM
from data_generator import generate_scenario
from simulated_annealing_solver import (
    simulated_annealing,
    cost_of_assignment,
    random_assignment,
)
from genetic_algorithm_solver import genetic_algorithm
from nsga2_solver import nsga2, latency_violation
from qaoa_solver import run_qaoa, run_exact


# (num_tasks, num_servers)
CONFIGS = [(10, 3), (25, 4), (50, 5), (100, 6)]
SEEDS = [1, 2, 3]
QAOA_CONFIGS = [(2, 2), (3, 2), (4, 2)]

METHODS = ["Random", "SA", "GA", "NSGA-II"]
COLORS = {"Random": "#B0B0B0", "SA": "#1F4E78", "GA": "#8E44AD", "NSGA-II": "#C0392B"}


def get_scenario(source, num_tasks, num_servers, seed):
    if source == "alibaba":
        return load_alibaba_scenario(num_tasks=num_tasks, num_servers=num_servers, seed=seed)
    scenario = generate_scenario(num_tasks=num_tasks, num_servers=num_servers, seed=seed)
    # Same rule as the Alibaba loader: total capacity = CAPACITY_HEADROOM x
    # total demand, so the instance is feasible and the constraint binds.
    total_demand = sum(t["size"] for t in scenario["tasks"])
    total_cap = sum(sv["capacity"] for sv in scenario["servers"])
    for sv in scenario["servers"]:
        share = sv["capacity"] / total_cap
        sv["capacity"] = int(math.ceil(CAPACITY_HEADROOM * total_demand * share))
    return scenario


def capacity_overflow(assignment, tasks, servers):
    """Total compute units assigned beyond server capacities (0 = feasible)."""
    size = {t["id"]: t["size"] for t in tasks}
    load = {s["id"]: 0 for s in servers}
    for task_id, server_id in assignment.items():
        load[server_id] += size[task_id]
    return sum(max(0, load[s["id"]] - s["capacity"]) for s in servers)


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def evaluate(method, assignment, tasks, servers, runtime, random_cost, meta):
    cost = cost_of_assignment(assignment, tasks, servers)
    improvement = (1 - cost / random_cost) * 100 if random_cost else 0.0
    return {
        **meta,
        "method": method,
        "cost": round(cost, 2),
        "latency_violation": round(latency_violation(assignment, tasks, servers), 2),
        "capacity_overflow": round(capacity_overflow(assignment, tasks, servers), 2),
        "runtime_s": round(runtime, 3),
        "improvement_vs_random_pct": round(improvement, 1),
    }


def run_classical(source):
    rows = []
    for num_tasks, num_servers in CONFIGS:
        for seed in SEEDS:
            scenario = get_scenario(source, num_tasks, num_servers, seed)
            tasks, servers = scenario["tasks"], scenario["servers"]
            meta = {"data": source, "num_tasks": num_tasks,
                    "num_servers": num_servers, "seed": seed}

            print(f"[{source}] {num_tasks} tasks / {num_servers} servers / seed {seed}")

            random.seed(seed)
            rand_assign = random_assignment(tasks, servers)
            random_cost = cost_of_assignment(rand_assign, tasks, servers)
            rows.append(evaluate("Random", rand_assign, tasks, servers, 0.0, random_cost, meta))

            (sa_a, _, _), sa_t = timed(simulated_annealing, tasks, servers, seed=seed)
            rows.append(evaluate("SA", sa_a, tasks, servers, sa_t, random_cost, meta))

            (ga_a, _, _), ga_t = timed(genetic_algorithm, tasks, servers, seed=seed)
            rows.append(evaluate("GA", ga_a, tasks, servers, ga_t, random_cost, meta))

            (_, ns_a, _, _, _), ns_t = timed(nsga2, tasks, servers, seed=seed)
            rows.append(evaluate("NSGA-II", ns_a, tasks, servers, ns_t, random_cost, meta))
    return rows


def brute_force_best(tasks, servers):
    """Exhaustive search using the SAME cost function as the other solvers."""
    start = time.perf_counter()
    ids = [s["id"] for s in servers]
    best_assign, best_cost = None, float("inf")
    for combo in itertools.product(ids, repeat=len(tasks)):
        assign = {t["id"]: sid for t, sid in zip(tasks, combo)}
        c = cost_of_assignment(assign, tasks, servers)
        if c < best_cost:
            best_assign, best_cost = assign, c
    return best_assign, best_cost, time.perf_counter() - start


def repair_capacity(assignment, tasks, servers, max_iters=200):
    """
    Classical repair step for the hybrid pipeline: greedily move tasks off
    overloaded servers (lowest overflow first, then lowest cost).
    """
    assign = dict(assignment)
    size = {t["id"]: t["size"] for t in tasks}
    cap = {s["id"]: s["capacity"] for s in servers}
    ids = [s["id"] for s in servers]

    for _ in range(max_iters):
        cur_over = capacity_overflow(assign, tasks, servers)
        if cur_over == 0:
            break
        load = {sid: 0 for sid in ids}
        for tid, sid in assign.items():
            load[sid] += size[tid]

        best = None
        for tid, sid in assign.items():
            if load[sid] <= cap[sid]:
                continue
            for target in ids:
                if target == sid:
                    continue
                trial = dict(assign)
                trial[tid] = target
                over = capacity_overflow(trial, tasks, servers)
                if over < cur_over:
                    key = (over, cost_of_assignment(trial, tasks, servers))
                    if best is None or key < best[0]:
                        best = (key, trial)
        if best is None:
            break
        assign = best[1]
    return assign


def brute_force_best(tasks, servers):
    """Exhaustive search using the SAME cost function as the other solvers."""
    start = time.perf_counter()
    ids = [s["id"] for s in servers]
    best_assign, best_cost = None, float("inf")
    for combo in itertools.product(ids, repeat=len(tasks)):
        assign = {t["id"]: sid for t, sid in zip(tasks, combo)}
        c = cost_of_assignment(assign, tasks, servers)
        if c < best_cost:
            best_assign, best_cost = assign, c
    return best_assign, best_cost, time.perf_counter() - start


def pct_gap(value, reference):
    return (value - reference) / abs(reference) * 100 if reference else 0.0


def run_qaoa_benchmark(source):
    """
    Three honest numbers per instance:
      gap_own_problem_pct : QAOA vs exact on the QUBO QAOA actually solves
      gap_real_world_pct  : QAOA answer scored on the full objective
                            (incl. capacity) vs the true optimum
      gap_repaired_pct    : QAOA + classical capacity repair (hybrid)
    """
    rows = []
    for num_tasks, num_servers in QAOA_CONFIGS:
        print(f"[QAOA] {num_tasks} tasks / {num_servers} servers")
        scenario = get_scenario(source, num_tasks, num_servers, seed=42)
        tasks, servers = scenario["tasks"], scenario["servers"]
        try:
            q_out = run_qaoa(tasks, servers, seed=1)
            e_out = run_exact(tasks, servers)
        except Exception as exc:
            print(f"   QAOA failed at this size: {exc}")
            continue

        q_assign, q_qubo, info = q_out[0], q_out[1], q_out[2]
        e_qubo, e_qubo_runtime = e_out[1], e_out[2]

        _, opt_cost, _ = brute_force_best(tasks, servers)
        q_cost = cost_of_assignment(q_assign, tasks, servers)
        rep_assign = repair_capacity(q_assign, tasks, servers)
        rep_cost = cost_of_assignment(rep_assign, tasks, servers)

        rows.append({
            "data": source,
            "num_tasks": num_tasks,
            "num_servers": num_servers,
            "qubits": info["num_qubits"],
            "gap_own_problem_pct": round(pct_gap(q_qubo, e_qubo), 2),
            "qaoa_cost": round(q_cost, 2),
            "qaoa_capacity_overflow": round(capacity_overflow(q_assign, tasks, servers), 2),
            "gap_real_world_pct": round(pct_gap(q_cost, opt_cost), 2),
            "repaired_cost": round(rep_cost, 2),
            "repaired_overflow": round(capacity_overflow(rep_assign, tasks, servers), 2),
            "gap_repaired_pct": round(pct_gap(rep_cost, opt_cost), 2),
            "optimal_cost": round(opt_cost, 2),
            "qaoa_runtime_s": round(info["runtime_seconds"], 3),
            "exact_runtime_s": round(e_qubo_runtime, 5),
        })
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def aggregate(rows, key):
    """Mean of `key` per (num_tasks, method)."""
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["num_tasks"], r["method"])].append(r[key])
    return {k: mean(v) for k, v in bucket.items()}


def print_summary(rows):
    cost = aggregate(rows, "cost")
    viol = aggregate(rows, "latency_violation")
    over = aggregate(rows, "capacity_overflow")
    rt = aggregate(rows, "runtime_s")
    print("\n" + "=" * 78)
    print("SUMMARY (mean over seeds)")
    print("=" * 78)
    print(f"{'Tasks':>5} | {'Method':8} | {'Cost':>9} | {'Lat.viol':>9} | {'Cap.over':>9} | {'Time(s)':>8}")
    print("-" * 78)
    for n, _ in CONFIGS:
        for m in METHODS:
            k = (n, m)
            if k in cost:
                print(f"{n:5d} | {m:8} | {cost[k]:9.2f} | {viol[k]:9.2f} | {over[k]:9.2f} | {rt[k]:8.2f}")
        print("-" * 78)


def plot_classical(rows):
    sizes = [n for n, _ in CONFIGS]
    cost = aggregate(rows, "cost")
    rt = aggregate(rows, "runtime_s")

    plt.figure(figsize=(9, 5))
    width = 0.2
    for j, m in enumerate(METHODS):
        xs = [i + (j - 1.5) * width for i in range(len(sizes))]
        plt.bar(xs, [cost.get((n, m), 0) for n in sizes], width, label=m, color=COLORS[m])
    plt.xticks(range(len(sizes)), [str(n) for n in sizes])
    plt.xlabel("Number of tasks")
    plt.ylabel("Mean total cost (lower = better)")
    plt.title("Solver cost vs problem size")
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_cost.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for m in ["SA", "GA", "NSGA-II"]:
        plt.plot(sizes, [rt.get((n, m), float("nan")) for n in sizes],
                 marker="o", label=m, color=COLORS[m], linewidth=2)
    plt.yscale("log")
    plt.xlabel("Number of tasks")
    plt.ylabel("Mean runtime (s, log scale)")
    plt.title("Solver runtime vs problem size")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_runtime.png", dpi=150)
    plt.close()
    print("Saved benchmark_cost.png, benchmark_runtime.png")


def plot_qaoa(rows):
    if not rows:
        return
    qubits = [r["qubits"] for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(qubits, [r["qaoa_runtime_s"] for r in rows], marker="o",
             color="#16A085", linewidth=2, label="QAOA (simulator)")
    plt.plot(qubits, [r["exact_runtime_s"] for r in rows], marker="s",
             color="#2C3E50", linewidth=2, label="Exact")
    plt.yscale("log")
    plt.xlabel("Qubits")
    plt.ylabel("Runtime (s, log scale)")
    plt.title("QAOA vs exact: runtime vs qubit count")
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qaoa_scaling.png", dpi=150)
    plt.close()
    print("Saved qaoa_scaling.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["alibaba", "synthetic"], default="alibaba")
    parser.add_argument("--skip-qaoa", action="store_true")
    args = parser.parse_args()

    rows = run_classical(args.data)
    write_csv("benchmark_results.csv", rows)
    print_summary(rows)
    plot_classical(rows)

    if not args.skip_qaoa:
        qrows = run_qaoa_benchmark(args.data)
        write_csv("qaoa_results.csv", qrows)
        if qrows:
            print("\nQAOA results:")
            for r in qrows:
                print(f"  {r['num_tasks']}T/{r['num_servers']}S  qubits={r['qubits']}  "
                      f"own-problem gap={r['gap_own_problem_pct']}%  "
                      f"overflow={r['qaoa_capacity_overflow']}  "
                      f"real-world gap={r['gap_real_world_pct']}%  "
                      f"repaired gap={r['gap_repaired_pct']}%  "
                      f"time={r['qaoa_runtime_s']}s")
        plot_qaoa(qrows)


if __name__ == "__main__":
    main()