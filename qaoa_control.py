"""
qaoa_control.py
----------------
Control experiment for the QAOA hybrid pipeline.

Question: does QAOA add anything beyond the classical capacity-repair step?

For every small instance (several scenarios per size) we start from five
different assignments, apply the SAME greedy capacity repair to each, and
compare the repaired result with the true optimum (brute force on the shared
cost function):

    QAOA            : QAOA's assignment-only answer, then repair
    Exact-QUBO      : exact solution of the SAME assignment-only QUBO
                      (the classical counterpart of QAOA), then repair
    Cheapest-server : everything on the lowest-energy server, then repair
    Random (mean)   : average over many random assignments, then repair
    Repair only     : n/a (repair needs a starting point) - covered above

Outputs:
    qaoa_control_<data>.csv   one row per (size, seed, method)
    qaoa_control_<data>.png   mean repaired gap per method and size

Usage:
    python qaoa_control.py                    # Alibaba data, 5 seeds
    python qaoa_control.py --data synthetic
    python qaoa_control.py --seeds 3
"""

import argparse
import csv
import random
import warnings
from collections import defaultdict
from statistics import mean

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from scipy.sparse import SparseEfficiencyWarning
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_benchmark import (
    get_scenario,
    capacity_overflow,
    brute_force_best,
    repair_capacity,
    pct_gap,
    QAOA_CONFIGS,
)
from simulated_annealing_solver import cost_of_assignment, random_assignment
from qaoa_solver import run_qaoa, run_exact

RANDOM_DRAWS = 20
METHODS = ["QAOA", "Exact-QUBO", "Cheapest-server", "Random (mean)"]
COLORS = {
    "QAOA": "#16A085",
    "Exact-QUBO": "#2C3E50",
    "Cheapest-server": "#E67E22",
    "Random (mean)": "#B0B0B0",
}


def cheapest_server_assignment(tasks, servers):
    best = min(servers, key=lambda s: s["energy_cost_per_unit"])["id"]
    return {t["id"]: best for t in tasks}


def score(assignment, tasks, servers, opt_cost):
    """Repair an assignment, then return (gap %, overflow, hit_optimum)."""
    repaired = repair_capacity(assignment, tasks, servers)
    cost = cost_of_assignment(repaired, tasks, servers)
    gap = pct_gap(cost, opt_cost)
    over = capacity_overflow(repaired, tasks, servers)
    return gap, over


def run_control(source, seeds):
    rows = []
    for num_tasks, num_servers in QAOA_CONFIGS:
        for seed in seeds:
            print(f"[{source}] {num_tasks}T/{num_servers}S  seed {seed}")
            scenario = get_scenario(source, num_tasks, num_servers, seed)
            tasks, servers = scenario["tasks"], scenario["servers"]
            opt_assign, opt_cost, _ = brute_force_best(tasks, servers)
            opt_over = capacity_overflow(opt_assign, tasks, servers)

            meta = {"data": source, "num_tasks": num_tasks,
                    "num_servers": num_servers, "seed": seed,
                    "optimum_overflow": round(opt_over, 2)}

            def add(method, gap, over):
                rows.append({
                    **meta,
                    "method": method,
                    "repaired_gap_pct": round(gap, 2),
                    "repaired_overflow": round(over, 2),
                    "hit_optimum": int(gap <= 0.01),
                })

            # QAOA (needs the Qiskit simulator, slowest step)
            try:
                q_out = run_qaoa(tasks, servers, seed=seed)
                add("QAOA", *score(q_out[0], tasks, servers, opt_cost))
            except Exception as exc:
                print(f"   QAOA failed: {exc}")

            # Classical counterpart: exact solution of the same QUBO
            e_out = run_exact(tasks, servers)
            add("Exact-QUBO", *score(e_out[0], tasks, servers, opt_cost))

            # Cheapest-server baseline
            add("Cheapest-server",
                *score(cheapest_server_assignment(tasks, servers),
                       tasks, servers, opt_cost))

            # Random starting points, averaged
            gaps, overs = [], []
            for k in range(RANDOM_DRAWS):
                random.seed(seed * 1000 + k)
                g, o = score(random_assignment(tasks, servers),
                             tasks, servers, opt_cost)
                gaps.append(g)
                overs.append(o)
            rows.append({
                **meta,
                "method": "Random (mean)",
                "repaired_gap_pct": round(mean(gaps), 2),
                "repaired_overflow": round(mean(overs), 2),
                "hit_optimum": round(
                    sum(1 for g, o in zip(gaps, overs)
                        if g <= 0.01) / RANDOM_DRAWS, 2),
            })
    return rows


def summarize(rows):
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["num_tasks"], r["method"])].append(r)

    print("\n" + "=" * 78)
    print("CONTROL SUMMARY (mean over scenarios; gap vs true optimum after repair)")
    print("=" * 78)
    print(f"{'Tasks':>5} | {'Method':16} | {'Mean gap %':>10} | "
          f"{'Hit optimum':>11} | {'Mean overflow':>13}")
    print("-" * 78)
    for n, _ in QAOA_CONFIGS:
        for m in METHODS:
            rs = bucket.get((n, m))
            if not rs:
                continue
            hits = sum(r["hit_optimum"] for r in rs)
            print(f"{n:5d} | {m:16} | "
                  f"{mean(r['repaired_gap_pct'] for r in rs):10.2f} | "
                  f"{hits:5.1f}/{len(rs):<5d} | "
                  f"{mean(r['repaired_overflow'] for r in rs):13.2f}")
        ex = [r for r in rows if r["num_tasks"] == n and r["method"] == "Exact-QUBO"]
        infeasible = sum(1 for r in ex if r["optimum_overflow"] > 0)
        print(f"      instances infeasible by construction (optimum itself "
              f"overflows): {infeasible}/{len(ex)}")
        print("-" * 78)


def plot(rows, path):
    sizes = [n for n, _ in QAOA_CONFIGS]
    bucket = defaultdict(list)
    for r in rows:
        bucket[(r["num_tasks"], r["method"])].append(r["repaired_gap_pct"])

    plt.figure(figsize=(9, 5))
    width = 0.2
    for j, m in enumerate(METHODS):
        xs = [i + (j - 1.5) * width for i in range(len(sizes))]
        ys = [mean(bucket[(n, m)]) if bucket.get((n, m)) else 0 for n in sizes]
        plt.bar(xs, ys, width, label=m, color=COLORS[m])
    plt.xticks(range(len(sizes)), [f"{n} tasks" for n in sizes])
    plt.ylabel("Mean gap to optimum after repair (%)")
    plt.title("QAOA control: repaired gap by starting assignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["alibaba", "synthetic"], default="alibaba")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    seeds = list(range(1, args.seeds + 1))
    rows = run_control(args.data, seeds)

    csv_path = f"qaoa_control_{args.data}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    summarize(rows)
    plot(rows, f"qaoa_control_{args.data}.png")


if __name__ == "__main__":
    main()