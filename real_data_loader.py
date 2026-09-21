"""
real_data_loader.py
---------------------
Loads the real Kaggle edge-computing task dataset (real_tasks.csv) and
converts it into the same task format used by data_generator.py, so it
plugs directly into simulated_annealing_solver.py (and later GA/NSGA-II/
QAOA) without changing any solver code.

The Kaggle dataset only contains TASK data (no server/edge-node data),
so servers are still generated synthetically here, but tasks now come
from real data.

CSV columns expected (exact header names):
    Task ID, Task Size (FLOPs), Deadline (ms), Data Size (MB),
    Latency (ms), Bandwidth (Mbps), Processing Power (FLOPs/sec),
    Energy Consumption (W)
"""

import pandas as pd
import random
from data_generator import generate_servers


def load_real_tasks(csv_path="real_tasks.csv", num_tasks=10, seed=42):
    """
    Load a random sample of `num_tasks` rows from the real dataset and
    convert them into the same dict format as generate_tasks() in
    data_generator.py: {"id", "size", "latency_requirement"}.

    Works with either a .csv or .xlsx/.xls file — the extension is
    detected automatically, so you don't need to worry about which
    format your dataset was saved/renamed as.

    Keep num_tasks SMALL (10-30) for now — QAOA later on will only work
    on small problem sizes, and it keeps demos fast.
    """
    if csv_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    if seed is not None:
        random.seed(seed)
    sample = df.sample(n=num_tasks, random_state=seed).reset_index(drop=True)

    tasks = []
    for _, row in sample.iterrows():
        # Scale Task Size (FLOPs, e.g. ~3e8) down to a small "size" unit
        # comparable to the synthetic generator's 5-30 range, so the
        # existing cost function / server capacities still make sense.
        scaled_size = row["Task Size (FLOPs)"] / 3.5e7  # tune divisor as needed

        # Use Latency (ms) directly as the "latency_requirement" —
        # lower Latency (ms) = stricter requirement, same convention as before
        latency_req = row["Latency (ms)"] / 10  # scaled down similarly

        tasks.append({
            "id": f"T{int(row['Task ID'])}",
            "size": round(scaled_size, 2),
            "latency_requirement": round(latency_req, 2),
            # extra real-world fields kept for later use (dashboard, paper)
            "data_size_mb": row["Data Size (MB)"],
            "bandwidth_mbps": row["Bandwidth (Mbps)"],
            "energy_watts": row["Energy Consumption (W)"],
        })

    return tasks


def load_real_scenario(csv_path="real_tasks.csv", num_tasks=10, num_servers=3, seed=42):
    """Convenience function: real tasks + synthetic servers, same shape
    as generate_scenario() in data_generator.py."""
    tasks = load_real_tasks(csv_path, num_tasks=num_tasks, seed=seed)
    servers = generate_servers(num_servers=num_servers, seed=seed)
    return {"tasks": tasks, "servers": servers}


if __name__ == "__main__":
    scenario = load_real_scenario(num_tasks=10, num_servers=3, seed=42)

    print("=== REAL TASKS (sampled from Kaggle dataset) ===")
    for t in scenario["tasks"]:
        print(t)

    print("\n=== SYNTHETIC SERVERS ===")
    for s in scenario["servers"]:
        print(s)
