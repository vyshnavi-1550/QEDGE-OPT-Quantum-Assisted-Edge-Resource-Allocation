"""
simulated_annealing_solver.py
------------------------------
A classical baseline solver for the QEDGE-OPT task allocation problem.

Problem: assign each task to exactly one server such that:
    - no server's total assigned task size exceeds its capacity
    - the total "cost" (energy cost + a penalty for overloading) is minimized

This is the "initial working module" for Review 1 (20% implementation).
Later, this same problem will also be solved with Genetic Algorithm,
NSGA-II, and QAOA (Qiskit) for comparison.

Run this file directly (after running data_generator.py, or it will
generate its own scenario) to see a solution printed out.
"""

import random
import math
from data_generator import generate_scenario


def cost_of_assignment(assignment, tasks, servers):
    """
    Compute the total cost of a given assignment.

    assignment: dict mapping task_id -> server_id
    Cost = sum(task.size * server.energy_cost_per_unit) for every task
           + a large penalty for every unit a server is over capacity
    Lower cost is better.
    """
    server_lookup = {s["id"]: s for s in servers}
    task_lookup = {t["id"]: t for t in tasks}

    load = {s["id"]: 0 for s in servers}
    total_energy_cost = 0.0

    for task_id, server_id in assignment.items():
        task = task_lookup[task_id]
        server = server_lookup[server_id]
        load[server_id] += task["size"]
        total_energy_cost += task["size"] * server["energy_cost_per_unit"]

    # Penalty for exceeding server capacity (heavily weighted so the
    # solver strongly prefers valid, non-overloaded solutions)
    PENALTY_WEIGHT = 50
    overload_penalty = 0
    for server in servers:
        overload = max(0, load[server["id"]] - server["capacity"])
        overload_penalty += overload * PENALTY_WEIGHT

    return total_energy_cost + overload_penalty


def random_assignment(tasks, servers):
    """Create a random starting assignment: each task -> a random server."""
    server_ids = [s["id"] for s in servers]
    return {t["id"]: random.choice(server_ids) for t in tasks}


def neighbor(assignment, tasks, servers):
    """Produce a neighboring solution by reassigning one random task."""
    new_assignment = dict(assignment)
    task_id = random.choice([t["id"] for t in tasks])
    server_ids = [s["id"] for s in servers]
    new_assignment[task_id] = random.choice(server_ids)
    return new_assignment


def simulated_annealing(tasks, servers, initial_temp=100.0, cooling_rate=0.98,
                         min_temp=0.1, iterations_per_temp=50, seed=None):
    """
    Run simulated annealing to find a low-cost task-to-server assignment.

    Returns: (best_assignment, best_cost, history)
    history is a list of (temperature, best_cost_so_far) for plotting later.
    """
    if seed is not None:
        random.seed(seed)

    current = random_assignment(tasks, servers)
    current_cost = cost_of_assignment(current, tasks, servers)

    best = dict(current)
    best_cost = current_cost

    temperature = initial_temp
    history = []

    while temperature > min_temp:
        for _ in range(iterations_per_temp):
            candidate = neighbor(current, tasks, servers)
            candidate_cost = cost_of_assignment(candidate, tasks, servers)

            delta = candidate_cost - current_cost
            # Accept better solutions always; accept worse ones sometimes
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current = candidate
                current_cost = candidate_cost

                if current_cost < best_cost:
                    best = dict(current)
                    best_cost = current_cost

        history.append((round(temperature, 3), best_cost))
        temperature *= cooling_rate

    return best, best_cost, history


if __name__ == "__main__":
    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)
    tasks = scenario["tasks"]
    servers = scenario["servers"]

    print("=== PROBLEM ===")
    print("Tasks:", tasks)
    print("Servers:", servers)

    best_assignment, best_cost, history = simulated_annealing(
        tasks, servers, seed=1
    )

    print("\n=== BEST ASSIGNMENT FOUND ===")
    for task_id, server_id in best_assignment.items():
        print(f"{task_id} -> {server_id}")

    print(f"\nBest cost (lower is better): {best_cost:.2f}")
    print(f"Number of cooling steps recorded: {len(history)}")
