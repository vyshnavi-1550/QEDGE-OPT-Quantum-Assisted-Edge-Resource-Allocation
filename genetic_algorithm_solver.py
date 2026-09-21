"""
genetic_algorithm_solver.py

Genetic Algorithm solver for the Quantum-Assisted Edge Resource Allocation
(QEDGE-OPT) task-to-server assignment problem.

Drop this file next to simulated_annealing_solver.py. It reuses
cost_of_assignment() and random_assignment() from simulated_annealing_solver.py
so that GA's cost is computed by the exact same function as SA's — this
guarantees a fair, apples-to-apples comparison between the two solvers.

Expects the same scenario format produced by data_generator.py:

    tasks   = [{"id": "T1", "size": 25, "latency_requirement": 2}, ...]
    servers = [{"id": "S1", "capacity": 42, "energy_cost_per_unit": 0.93}, ...]

Usage:
    from data_generator import generate_scenario
    from genetic_algorithm_solver import genetic_algorithm

    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)
    best_assignment, best_cost, history = genetic_algorithm(
        scenario["tasks"], scenario["servers"], seed=1
    )
"""

import random

from simulated_annealing_solver import cost_of_assignment, random_assignment


def _individual_to_assignment(individual, tasks, servers):
    """Convert a raw index list (one server index per task) into the
    {task_id: server_id} dict format used by cost_of_assignment()."""
    return {
        tasks[task_idx]["id"]: servers[server_idx]["id"]
        for task_idx, server_idx in enumerate(individual)
    }


def genetic_algorithm(
    tasks,
    servers,
    population_size=60,
    generations=200,
    crossover_rate=0.85,
    mutation_rate=0.15,
    tournament_size=3,
    elitism_count=2,
    seed=None,
    verbose=False,
):
    """
    Run a Genetic Algorithm over task-to-server assignments.

    Returns:
        best_assignment (dict): {task_id: server_id}
        best_cost (float): cost_of_assignment() value for best_assignment
        history (list of (generation, cost) tuples): best cost per generation,
            same shape as simulated_annealing()'s history so demo.py's
            plotting code can be reused with generation in place of temperature.
    """
    if seed is not None:
        random.seed(seed)

    num_tasks = len(tasks)
    num_servers = len(servers)

    def random_individual():
        return [random.randrange(num_servers) for _ in range(num_tasks)]

    def fitness(individual):
        assignment = _individual_to_assignment(individual, tasks, servers)
        return cost_of_assignment(assignment, tasks, servers)

    def tournament_select(population, fitnesses):
        contenders = random.sample(range(len(population)), tournament_size)
        best_idx = min(contenders, key=lambda i: fitnesses[i])
        return population[best_idx]

    def crossover(parent1, parent2):
        if random.random() > crossover_rate or num_tasks < 2:
            return parent1[:], parent2[:]
        point = random.randint(1, num_tasks - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(individual):
        for i in range(num_tasks):
            if random.random() < mutation_rate:
                individual[i] = random.randrange(num_servers)
        return individual

    population = [random_individual() for _ in range(population_size)]
    history = []
    best_individual = None
    best_fitness = float("inf")

    for gen in range(generations):
        fitnesses = [fitness(ind) for ind in population]

        gen_best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_individual = population[gen_best_idx][:]

        history.append((gen, best_fitness))

        if verbose and gen % 20 == 0:
            print(f"Generation {gen:4d} | Best cost: {best_fitness:.2f}")

        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        new_population = [population[i][:] for i in ranked[:elitism_count]]

        while len(new_population) < population_size:
            parent1 = tournament_select(population, fitnesses)
            parent2 = tournament_select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            if len(new_population) < population_size:
                new_population.append(mutate(child2))

        population = new_population

    best_assignment = _individual_to_assignment(best_individual, tasks, servers)
    return best_assignment, best_fitness, history



def plot_convergence(history, filepath="ga_progress.png"):
    """Save a convergence chart, mirroring annealing_progress.png from the SA solver."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generations = [h[0] for h in history]
    costs = [h[1] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(generations, costs, color="#1F4E78", linewidth=2)
    plt.xlabel("Generation")
    plt.ylabel("Best cost found so far")
    plt.title("Genetic Algorithm: Cost Improving Over Time")
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

    best_assignment, best_cost, history = genetic_algorithm(
        tasks, servers, seed=1, verbose=True
    )

    print("\n--- RESULT: NAIVE (RANDOM) VS OPTIMIZED (GENETIC ALGORITHM) ---")
    print(f"Random assignment    : {naive_cost:.2f}")
    print(f"Genetic Algorithm    : {best_cost:.2f}")
    improvement = (1 - best_cost / naive_cost) * 100 if naive_cost else 0
    print(f"Improvement over naive baseline: {improvement:.1f}% lower cost")

    print("\nBest assignment found:")
    for task_id, server_id in best_assignment.items():
        print(f"  {task_id} -> {server_id}")

    plot_convergence(history)
