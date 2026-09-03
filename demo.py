"""
demo.py
--------
Presentation-friendly demo of the QEDGE-OPT "initial working module"
for Review 1.

What it does:
    1. Generates a synthetic scenario (tasks + edge servers)
    2. Shows the problem clearly in table form
    3. Runs Simulated Annealing to solve it
    4. Compares a random (naive) assignment vs. the optimized assignment
    5. Saves a chart showing how the solution improves as the algorithm runs

Run this with:  python3 demo.py
"""

import matplotlib
matplotlib.use("Agg")  # no display needed, just save the image
import matplotlib.pyplot as plt

from data_generator import generate_scenario
from simulated_annealing_solver import (
    simulated_annealing,
    cost_of_assignment,
    random_assignment,
)


def print_table(headers, rows):
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))))


def main():
    print("=" * 60)
    print("QEDGE-OPT — Review 1 Demo: Simulated Annealing Baseline")
    print("=" * 60)

    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)
    tasks = scenario["tasks"]
    servers = scenario["servers"]

    print("\n--- TASKS ---")
    print_table(
        ["Task ID", "Size", "Latency Req."],
        [[t["id"], t["size"], t["latency_requirement"]] for t in tasks],
    )

    print("\n--- SERVERS ---")
    print_table(
        ["Server ID", "Capacity", "Energy Cost/Unit"],
        [[s["id"], s["capacity"], s["energy_cost_per_unit"]] for s in servers],
    )

    # Baseline: a random, naive assignment (what you'd get with no optimization)
    naive = random_assignment(tasks, servers)
    naive_cost = cost_of_assignment(naive, tasks, servers)

    # Optimized: simulated annealing result
    best_assignment, best_cost, history = simulated_annealing(tasks, servers, seed=1)

    print("\n--- RESULT: NAIVE (RANDOM) VS OPTIMIZED (SIMULATED ANNEALING) ---")
    print_table(
        ["Method", "Total Cost (lower = better)"],
        [
            ["Random assignment", f"{naive_cost:.2f}"],
            ["Simulated Annealing", f"{best_cost:.2f}"],
        ],
    )
    improvement = (1 - best_cost / naive_cost) * 100 if naive_cost else 0
    print(f"\nImprovement over naive baseline: {improvement:.1f}% lower cost")

    print("\n--- FINAL TASK -> SERVER ASSIGNMENT ---")
    print_table(
        ["Task ID", "Assigned Server"],
        [[task_id, server_id] for task_id, server_id in best_assignment.items()],
    )

    # Save a chart showing cost improving over the annealing process
    temps = [h[0] for h in history]
    costs = [h[1] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(costs)), costs, color="#1F4E78", linewidth=2)
    plt.xlabel("Cooling step")
    plt.ylabel("Best cost found so far")
    plt.title("Simulated Annealing: Cost Improving Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("annealing_progress.png", dpi=150)
    print("\nChart saved as annealing_progress.png (use this in your PPT)")


if __name__ == "__main__":
    main()
