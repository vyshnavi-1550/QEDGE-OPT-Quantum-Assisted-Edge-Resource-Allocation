"""
nsga2_solver.py

NSGA-II (Non-dominated Sorting Genetic Algorithm II) solver for QEDGE-OPT.

Extends the single-objective GA to TWO objectives, both minimized:
  1. Total cost — reuses cost_of_assignment() from simulated_annealing_solver.py
     (energy cost + capacity-violation penalty), the exact same function used by
     SA and GA, so all three solvers remain directly comparable on cost.
  2. Total latency-requirement violation — how much each task's assigned server's
     estimated latency exceeds that task's latency_requirement, summed across all
     tasks. Server latency is modeled as increasing with load (congestion), using
     only the existing "capacity" field — no changes to data_generator.py needed.

Returns a Pareto front (the set of non-dominated trade-off solutions) plus a
single "best compromise" pick for easy comparison against SA/GA's single number.

Usage:
    from data_generator import generate_scenario
    from nsga2_solver import nsga2

    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)
    pareto_front, best_assignment, best_cost, best_violation, history = nsga2(
        scenario["tasks"], scenario["servers"], seed=1
    )
"""

import random

from simulated_annealing_solver import cost_of_assignment, random_assignment

BASE_LATENCY = 1.0       # ms — minimum latency of an unloaded server
CONGESTION_FACTOR = 4.0  # ms — extra delay added at 100% utilization


def _individual_to_assignment(individual, tasks, servers):
    return {tasks[i]["id"]: servers[s]["id"] for i, s in enumerate(individual)}


def _estimated_latency(server, load):
    capacity = server["capacity"] if server["capacity"] > 0 else 1
    utilization = min(load / capacity, 2.0)  # cap extreme overload for stability
    return BASE_LATENCY + CONGESTION_FACTOR * utilization


def latency_violation(assignment, tasks, servers):
    """Sum of (estimated_latency - required_latency) over every task whose
    assigned server's estimated latency exceeds its latency_requirement."""
    server_map = {s["id"]: s for s in servers}
    load = {s["id"]: 0 for s in servers}
    for t in tasks:
        load[assignment[t["id"]]] += t["size"]

    violation = 0.0
    for t in tasks:
        server = server_map[assignment[t["id"]]]
        est_latency = _estimated_latency(server, load[assignment[t["id"]]])
        required = t.get("latency_requirement", t.get("latency_req", 0))
        if est_latency > required:
            violation += est_latency - required
    return violation


def _objectives(individual, tasks, servers):
    assignment = _individual_to_assignment(individual, tasks, servers)
    cost = cost_of_assignment(assignment, tasks, servers)
    violation = latency_violation(assignment, tasks, servers)
    return cost, violation


def _dominates(obj_a, obj_b):
    not_worse = all(a <= b for a, b in zip(obj_a, obj_b))
    strictly_better = any(a < b for a, b in zip(obj_a, obj_b))
    return not_worse and strictly_better


def _fast_non_dominated_sort(objectives_list):
    n = len(objectives_list)
    domination_counts = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    fronts = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objectives_list[p], objectives_list[q]):
                dominated_solutions[p].append(q)
            elif _dominates(objectives_list[q], objectives_list[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    fronts.pop()
    return fronts


def _crowding_distance(front, objectives_list):
    distance = {i: 0.0 for i in front}
    num_objectives = len(objectives_list[0])

    for m in range(num_objectives):
        front_sorted = sorted(front, key=lambda i: objectives_list[i][m])
        distance[front_sorted[0]] = float("inf")
        distance[front_sorted[-1]] = float("inf")
        obj_min = objectives_list[front_sorted[0]][m]
        obj_max = objectives_list[front_sorted[-1]][m]
        if obj_max == obj_min:
            continue
        for k in range(1, len(front_sorted) - 1):
            prev_obj = objectives_list[front_sorted[k - 1]][m]
            next_obj = objectives_list[front_sorted[k + 1]][m]
            distance[front_sorted[k]] += (next_obj - prev_obj) / (obj_max - obj_min)
    return distance


def _tournament_select(population, ranks, crowding, k=2):
    contenders = random.sample(range(len(population)), k)
    best = contenders[0]
    for c in contenders[1:]:
        if ranks[c] < ranks[best] or (ranks[c] == ranks[best] and crowding[c] > crowding[best]):
            best = c
    return population[best]


def nsga2(
    tasks,
    servers,
    population_size=60,
    generations=200,
    crossover_rate=0.85,
    mutation_rate=0.15,
    seed=None,
    verbose=False,
):
    """
    Run NSGA-II over task-to-server assignments, minimizing (cost, latency_violation).

    Returns:
        pareto_front (list of dicts): each {"assignment", "cost", "latency_violation"}
        best_assignment (dict): the chosen "best compromise" assignment
        best_cost (float)
        best_violation (float)
        history (list of (generation, best_cost_in_front0) tuples)
    """
    if seed is not None:
        random.seed(seed)

    num_tasks = len(tasks)
    num_servers = len(servers)

    def random_individual():
        return [random.randrange(num_servers) for _ in range(num_tasks)]

    def crossover(p1, p2):
        if random.random() > crossover_rate or num_tasks < 2:
            return p1[:], p2[:]
        point = random.randint(1, num_tasks - 1)
        return p1[:point] + p2[point:], p2[:point] + p1[point:]

    def mutate(ind):
        for i in range(num_tasks):
            if random.random() < mutation_rate:
                ind[i] = random.randrange(num_servers)
        return ind

    population = [random_individual() for _ in range(population_size)]
    history = []

    for gen in range(generations):
        objectives_list = [_objectives(ind, tasks, servers) for ind in population]
        fronts = _fast_non_dominated_sort(objectives_list)

        ranks = {}
        for rank, front in enumerate(fronts):
            for i in front:
                ranks[i] = rank

        crowding = {}
        for front in fronts:
            crowding.update(_crowding_distance(front, objectives_list))

        best_cost_gen = min(objectives_list[i][0] for i in fronts[0])
        history.append((gen, best_cost_gen))
        if verbose and gen % 20 == 0:
            print(f"Generation {gen:4d} | Front-0 size: {len(fronts[0]):3d} | Best cost in front: {best_cost_gen:.2f}")

        offspring = []
        while len(offspring) < population_size:
            p1 = _tournament_select(population, ranks, crowding)
            p2 = _tournament_select(population, ranks, crowding)
            c1, c2 = crossover(p1, p2)
            offspring.append(mutate(c1))
            if len(offspring) < population_size:
                offspring.append(mutate(c2))

        combined = population + offspring
        combined_objectives = objectives_list + [_objectives(ind, tasks, servers) for ind in offspring]
        combined_fronts = _fast_non_dominated_sort(combined_objectives)

        new_population = []
        for front in combined_fronts:
            if len(new_population) + len(front) <= population_size:
                new_population.extend(combined[i] for i in front)
            else:
                cd = _crowding_distance(front, combined_objectives)
                front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                remaining = population_size - len(new_population)
                new_population.extend(combined[i] for i in front_sorted[:remaining])
                break
        population = new_population

    final_objectives = [_objectives(ind, tasks, servers) for ind in population]
    final_fronts = _fast_non_dominated_sort(final_objectives)
    pareto_indices = final_fronts[0]

    pareto_front = []
    for i in pareto_indices:
        assignment = _individual_to_assignment(population[i], tasks, servers)
        cost, violation = final_objectives[i]
        pareto_front.append({"assignment": assignment, "cost": cost, "latency_violation": violation})

    # "Best compromise": lowest cost among zero-violation solutions;
    # falls back to lowest cost overall if none are fully feasible.
    feasible = [p for p in pareto_front if p["latency_violation"] == 0]
    pool = feasible if feasible else pareto_front
    best = min(pool, key=lambda p: p["cost"])

    return pareto_front, best["assignment"], best["cost"], best["latency_violation"], history


def plot_pareto_front(pareto_front, filepath="nsga2_pareto.png"):
    """Scatter plot of the final Pareto front: cost vs. latency violation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    costs = [p["cost"] for p in pareto_front]
    violations = [p["latency_violation"] for p in pareto_front]

    plt.figure(figsize=(8, 5))
    plt.scatter(costs, violations, color="#8E44AD", s=60, edgecolors="white", linewidths=0.8)
    plt.xlabel("Total Cost")
    plt.ylabel("Total Latency Violation")
    plt.title("NSGA-II: Final Pareto Front (Cost vs. Latency Violation)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Chart saved as {filepath}")


def plot_convergence(history, filepath="nsga2_progress.png"):
    """Best cost within the front-0 Pareto set, per generation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations = [h[0] for h in history]
    costs = [h[1] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(generations, costs, color="#8E44AD", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best cost in Pareto front")
    plt.title("NSGA-II: Best Cost in Front Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    print(f"Chart saved as {filepath}")


if __name__ == "__main__":
    from data_generator import generate_scenario

    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)
    tasks = scenario["tasks"]
    servers = scenario["servers"]

    naive = random_assignment(tasks, servers)
    naive_cost = cost_of_assignment(naive, tasks, servers)
    naive_violation = latency_violation(naive, tasks, servers)

    pareto_front, best_assignment, best_cost, best_violation, history = nsga2(
        tasks, servers, seed=1, verbose=True
    )

    print("\n--- RESULT: RANDOM vs NSGA-II (BEST COMPROMISE) ---")
    print(f"Random assignment      : cost={naive_cost:.2f}, latency violation={naive_violation:.2f}")
    print(f"NSGA-II best compromise: cost={best_cost:.2f}, latency violation={best_violation:.2f}")

    print(f"\nFinal Pareto front has {len(pareto_front)} trade-off solutions:")
    for p in sorted(pareto_front, key=lambda x: x["cost"]):
        print(f"  cost={p['cost']:7.2f}  |  latency violation={p['latency_violation']:6.2f}")

    print("\nBest compromise assignment:")
    for task_id, server_id in best_assignment.items():
        print(f"  {task_id} -> {server_id}")

    plot_convergence(history)
    plot_pareto_front(pareto_front)
