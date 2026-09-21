"""
demo.py
--------
Presentation-friendly demo of the QEDGE-OPT working modules.

What it does:
    1. Generates a synthetic scenario (tasks + edge servers)
    2. Shows the problem clearly in table form
    3. Runs Simulated Annealing (SA), Genetic Algorithm (GA), and NSGA-II to solve it
    4. Compares random (naive) vs. SA vs. GA vs. NSGA-II assignments
    5. Saves charts showing how each solution improves, plus NSGA-II's Pareto front

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
from genetic_algorithm_solver import genetic_algorithm
from nsga2_solver import nsga2, latency_violation, plot_pareto_front


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
    print("QEDGE-OPT — Demo: SA vs GA vs NSGA-II")
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
    naive_violation = latency_violation(naive, tasks, servers)

    # Simulated Annealing
    sa_assignment, sa_cost, sa_history = simulated_annealing(tasks, servers, seed=1)

    # Genetic Algorithm
    ga_assignment, ga_cost, ga_history = genetic_algorithm(tasks, servers, seed=1)

    # NSGA-II (multi-objective: cost + latency violation)
    pareto_front, nsga_assignment, nsga_cost, nsga_violation, nsga_history = nsga2(
        tasks, servers, seed=1
    )

    print("\n--- RESULT: NAIVE VS SIMULATED ANNEALING VS GENETIC ALGORITHM VS NSGA-II ---")
    print_table(
        ["Method", "Total Cost (lower = better)", "Latency Violation (lower = better)"],
        [
            ["Random assignment", f"{naive_cost:.2f}", f"{naive_violation:.2f}"],
            ["Simulated Annealing", f"{sa_cost:.2f}", "not optimized"],
            ["Genetic Algorithm", f"{ga_cost:.2f}", "not optimized"],
            ["NSGA-II (best compromise)", f"{nsga_cost:.2f}", f"{nsga_violation:.2f}"],
        ],
    )
    sa_improvement = (1 - sa_cost / naive_cost) * 100 if naive_cost else 0
    ga_improvement = (1 - ga_cost / naive_cost) * 100 if naive_cost else 0
    nsga_improvement = (1 - nsga_cost / naive_cost) * 100 if naive_cost else 0
    nsga_violation_improvement = (1 - nsga_violation / naive_violation) * 100 if naive_violation else 0
    print(f"\nSA improvement over naive baseline: {sa_improvement:.1f}% lower cost")
    print(f"GA improvement over naive baseline: {ga_improvement:.1f}% lower cost")
    print(f"NSGA-II improvement over naive baseline: {nsga_improvement:.1f}% lower cost, "
          f"{nsga_violation_improvement:.1f}% lower latency violation")
    print(f"NSGA-II Pareto front size: {len(pareto_front)} trade-off solutions")

    print("\n--- FINAL TASK -> SERVER ASSIGNMENT (Simulated Annealing) ---")
    print_table(
        ["Task ID", "Assigned Server"],
        [[task_id, server_id] for task_id, server_id in sa_assignment.items()],
    )

    print("\n--- FINAL TASK -> SERVER ASSIGNMENT (Genetic Algorithm) ---")
    print_table(
        ["Task ID", "Assigned Server"],
        [[task_id, server_id] for task_id, server_id in ga_assignment.items()],
    )

    print("\n--- FINAL TASK -> SERVER ASSIGNMENT (NSGA-II best compromise) ---")
    print_table(
        ["Task ID", "Assigned Server"],
        [[task_id, server_id] for task_id, server_id in nsga_assignment.items()],
    )

    # Chart 1: SA cost improving over the annealing process
    sa_costs = [h[1] for h in sa_history]
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(sa_costs)), sa_costs, color="#1F4E78", linewidth=2)
    plt.xlabel("Cooling step")
    plt.ylabel("Best cost found so far")
    plt.title("Simulated Annealing: Cost Improving Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("annealing_progress.png", dpi=150)
    print("\nChart saved as annealing_progress.png (use this in your PPT)")

    # Chart 2: GA cost improving over generations
    ga_gens = [h[0] for h in ga_history]
    ga_costs = [h[1] for h in ga_history]
    plt.figure(figsize=(8, 5))
    plt.plot(ga_gens, ga_costs, color="#8E44AD", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best cost found so far")
    plt.title("Genetic Algorithm: Cost Improving Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("ga_progress.png", dpi=150)
    print("Chart saved as ga_progress.png (use this in your PPT)")

    # Chart 3: NSGA-II convergence (best cost within front-0)
    nsga_gens = [h[0] for h in nsga_history]
    nsga_costs = [h[1] for h in nsga_history]
    plt.figure(figsize=(8, 5))
    plt.plot(nsga_gens, nsga_costs, color="#C0392B", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best cost in Pareto front")
    plt.title("NSGA-II: Best Cost in Front Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("nsga2_progress.png", dpi=150)
    print("Chart saved as nsga2_progress.png (use this in your PPT)")

    # Chart 4: NSGA-II Pareto front (cost vs latency violation)
    plot_pareto_front(pareto_front, filepath="nsga2_pareto.png")

    # Chart 5: side-by-side bar comparison of cost across all methods
    plt.figure(figsize=(6, 5))
    methods = ["Random", "Simulated\nAnnealing", "Genetic\nAlgorithm", "NSGA-II\n(compromise)"]
    values = [naive_cost, sa_cost, ga_cost, nsga_cost]
    colors = ["#B0B0B0", "#1F4E78", "#8E44AD", "#C0392B"]
    plt.bar(methods, values, color=colors)
    plt.ylabel("Total Cost (lower = better)")
    plt.title("Method Comparison: Total Cost")
    plt.tight_layout()
    plt.savefig("method_comparison.png", dpi=150)
    print("Chart saved as method_comparison.png (use this in your PPT)")


if __name__ == "__main__":
    main()
