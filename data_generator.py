"""
data_generator.py
------------------
Generates a synthetic edge-computing scenario: a list of tasks and a
set of edge servers, for use by all solvers (Simulated Annealing,
Genetic Algorithm, NSGA-II, QAOA) in the QEDGE-OPT project.

Each task has:
    - id
    - size (compute demand, arbitrary units)
    - latency_requirement (max acceptable delay)

Each server has:
    - id
    - capacity (total compute units it can handle)
    - energy_cost_per_unit (cost of processing one unit of task size)

Run this file directly to see an example scenario printed out.
"""

import random
import json


def generate_tasks(num_tasks=10, min_size=5, max_size=30, seed=None):
    """Create a list of synthetic tasks."""
    if seed is not None:
        random.seed(seed)

    tasks = []
    for i in range(num_tasks):
        task = {
            "id": f"T{i+1}",
            "size": random.randint(min_size, max_size),
            "latency_requirement": random.randint(1, 10),  # lower = stricter
        }
        tasks.append(task)
    return tasks


def generate_servers(num_servers=3, min_capacity=40, max_capacity=80, seed=None):
    """Create a list of synthetic edge servers."""
    if seed is not None:
        random.seed(seed + 1 if seed is not None else None)

    servers = []
    for j in range(num_servers):
        server = {
            "id": f"S{j+1}",
            "capacity": random.randint(min_capacity, max_capacity),
            "energy_cost_per_unit": round(random.uniform(0.5, 2.0), 2),
        }
        servers.append(server)
    return servers


def generate_scenario(num_tasks=10, num_servers=3, seed=42):
    """Convenience function: generate a full scenario (tasks + servers)."""
    tasks = generate_tasks(num_tasks=num_tasks, seed=seed)
    servers = generate_servers(num_servers=num_servers, seed=seed)
    return {"tasks": tasks, "servers": servers}


def save_scenario(scenario, filepath="scenario.json"):
    with open(filepath, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"Scenario saved to {filepath}")


if __name__ == "__main__":
    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)

    print("=== TASKS ===")
    for t in scenario["tasks"]:
        print(t)

    print("\n=== SERVERS ===")
    for s in scenario["servers"]:
        print(s)

    save_scenario(scenario, "scenario.json")
