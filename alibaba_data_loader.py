"""
alibaba_data_loader.py

Loads Alibaba Cluster Trace v2017 and converts a sample of it into the
same task/server format produced by data_generator.py, so every solver
(SA, GA, NSGA-II, QAOA) can run on it unchanged.

Files used (inside alibaba-trace-2017/):
    batch_task.csv
    server_event.csv

What comes from the real trace:
    - task workload      (plan_cpu x instance_num = total cores requested)
    - relative task demand ordering (used for latency ranking)
    - relative server CPU capacity (cpu_capacity)

What is derived / synthetic (state this in the paper):
    - task size is log-scaled to the integer range 5-30 (same range as
      the synthetic generator) because raw trace demand is heavy-tailed
    - latency_requirement: rank-based from demand, integer 1-10
      (heavier tasks get stricter deadlines)
    - server capacity is scaled so total capacity = CAPACITY_HEADROOM x
      total task demand, so the capacity constraint actually binds
    - energy_cost_per_unit: evenly spread between 0.5 and 2.0
"""

import math
import os
from functools import lru_cache

import numpy as np
import pandas as pd


BASE_DIR = "alibaba-trace-2017"

BATCH_TASK_FILE = os.path.join(BASE_DIR, "batch_task.csv")
SERVER_EVENT_FILE = os.path.join(BASE_DIR, "server_event.csv")

# Alibaba v2017 plan_cpu is in hundredths of a core (100 = 1 core).
CPU_UNIT_DIVISOR = 100.0

# Rows of batch_task.csv read before sampling.
ROWS_TO_READ = 200000

SEED = 42

# Same ranges as data_generator.py
SIZE_RANGE = (5, 30)
LATENCY_RANGE = (1, 10)

# Total server capacity as a multiple of total task demand.
CAPACITY_HEADROOM = 1.4


@lru_cache(maxsize=1)
def _read_batch_tasks():
    if not os.path.exists(BATCH_TASK_FILE):
        raise FileNotFoundError(f"Could not find {BATCH_TASK_FILE}")

    columns = [
        "start_time", "end_time", "job_id", "task_id",
        "instance_num", "status", "plan_cpu", "plan_mem",
    ]
    df = pd.read_csv(
        BATCH_TASK_FILE, header=None, names=columns, nrows=ROWS_TO_READ
    )
    for col in ["plan_cpu", "plan_mem", "job_id", "task_id", "instance_num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["plan_cpu", "plan_mem", "job_id", "task_id", "instance_num"]
    )
    df = df[(df["plan_cpu"] > 0) & (df["instance_num"] > 0)].copy()
    return df


@lru_cache(maxsize=1)
def _read_server_events():
    if not os.path.exists(SERVER_EVENT_FILE):
        raise FileNotFoundError(f"Could not find {SERVER_EVENT_FILE}")

    columns = [
        "timestamp", "machine_id", "event_type", "event_detail",
        "cpu_capacity", "memory_capacity", "disk_capacity",
    ]
    df = pd.read_csv(SERVER_EVENT_FILE, header=None, names=columns)
    df = df[df["event_type"] == "add"].copy()
    df["cpu_capacity"] = pd.to_numeric(df["cpu_capacity"], errors="coerce")
    df = df.dropna(subset=["cpu_capacity"])
    df = df.drop_duplicates(subset=["machine_id"])
    return df


def load_alibaba_tasks(num_tasks=10, seed=SEED):
    """Reproducible random sample of tasks in QEDGE-OPT format."""

    pool = _read_batch_tasks()
    if len(pool) < num_tasks:
        raise ValueError(
            f"Only {len(pool)} valid tasks available, {num_tasks} requested. "
            f"Increase ROWS_TO_READ."
        )

    df = pool.sample(n=num_tasks, random_state=seed).reset_index(drop=True)

    # Total cores requested by the task across its instances.
    df["raw_demand"] = (df["plan_cpu"] / CPU_UNIT_DIVISOR) * df["instance_num"]

    # Log-scale to SIZE_RANGE (heavy-tailed raw demand).
    logd = np.log1p(df["raw_demand"].to_numpy(dtype=float))
    lo, hi = SIZE_RANGE
    if logd.max() > logd.min():
        scaled = lo + (logd - logd.min()) / (logd.max() - logd.min()) * (hi - lo)
    else:
        scaled = np.full(len(logd), (lo + hi) / 2.0)
    df["size"] = np.rint(scaled).astype(int)

    # Rank-based latency: heavier tasks -> stricter (lower) requirement.
    llo, lhi = LATENCY_RANGE
    if num_tasks > 1:
        pct = df["raw_demand"].rank(pct=True, method="first")
        lat = llo + (lhi - llo) * (1 - pct)
    else:
        lat = pd.Series([(llo + lhi) / 2.0])
    df["latency_requirement"] = np.rint(lat).astype(int)

    tasks = []
    for i, row in df.iterrows():
        tasks.append({
            "id": f"T{i + 1}",
            "size": int(row["size"]),
            "latency_requirement": int(row["latency_requirement"]),

            # Original Alibaba information (kept for the paper)
            "job_id": int(row["job_id"]),
            "task_id": int(row["task_id"]),
            "instance_num": int(row["instance_num"]),
            "plan_cpu": float(row["plan_cpu"]),
            "plan_mem": float(row["plan_mem"]),
            "raw_demand_cores": round(float(row["raw_demand"]), 2),
        })
    return tasks


def load_alibaba_servers(num_servers=3):
    """Servers with raw (unscaled) Alibaba CPU capacity."""

    df = _read_server_events().head(num_servers)
    if len(df) < num_servers:
        raise ValueError(
            f"Only {len(df)} servers available, {num_servers} requested."
        )

    servers = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        servers.append({
            "id": f"S{i}",
            "raw_cpu_capacity": float(row["cpu_capacity"]),
            "machine_id": int(row["machine_id"]),
        })
    return servers


def load_alibaba_scenario(num_tasks=10, num_servers=3, seed=SEED,
                          capacity_headroom=CAPACITY_HEADROOM):
    """
    Full scenario in the same format as data_generator.generate_scenario():
        tasks:   id, size, latency_requirement
        servers: id, capacity, energy_cost_per_unit
    (extra Alibaba fields are kept and ignored by the solvers).
    """

    tasks = load_alibaba_tasks(num_tasks=num_tasks, seed=seed)
    raw_servers = load_alibaba_servers(num_servers=num_servers)

    total_demand = sum(t["size"] for t in tasks)
    total_raw = sum(s["raw_cpu_capacity"] for s in raw_servers)

    servers = []
    for i, s in enumerate(raw_servers):
        share = s["raw_cpu_capacity"] / total_raw
        capacity = int(math.ceil(capacity_headroom * total_demand * share))

        if num_servers > 1:
            energy = 0.5 + 1.5 * i / (num_servers - 1)
        else:
            energy = 1.0

        servers.append({
            "id": s["id"],
            "capacity": capacity,
            "energy_cost_per_unit": round(energy, 2),
            "machine_id": s["machine_id"],
            "raw_cpu_capacity": s["raw_cpu_capacity"],
        })

    return {"tasks": tasks, "servers": servers}


if __name__ == "__main__":

    print("=" * 60)
    print("QEDGE-OPT - Alibaba Cluster Trace v2017")
    print("=" * 60)

    scenario = load_alibaba_scenario(num_tasks=10, num_servers=3)

    print("\n--- ALIBABA TASKS ---")
    print("Task ID | Size | Latency Req. | Raw cores | Instances")
    print("-" * 60)
    for t in scenario["tasks"]:
        print(
            f"{t['id']:7} | {t['size']:4d} | {t['latency_requirement']:12d} | "
            f"{t['raw_demand_cores']:9.2f} | {t['instance_num']:9d}"
        )

    print("\n--- ALIBABA SERVERS (capacity scaled) ---")
    print("Server | Capacity | Energy/Unit | Raw CPU | Machine ID")
    print("-" * 60)
    for s in scenario["servers"]:
        print(
            f"{s['id']:6} | {s['capacity']:8d} | "
            f"{s['energy_cost_per_unit']:11.2f} | "
            f"{s['raw_cpu_capacity']:7.1f} | {s['machine_id']}"
        )

    demand = sum(t["size"] for t in scenario["tasks"])
    cap = sum(s["capacity"] for s in scenario["servers"])
    print(f"\nTotal demand: {demand}   Total capacity: {cap}")
    print("Alibaba data loading successful.")
