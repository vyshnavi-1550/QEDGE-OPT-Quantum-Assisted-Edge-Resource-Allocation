"""
qubo_formulation.py

QUBO (Quadratic Unconstrained Binary Optimization) formulation of the
QEDGE-OPT task-to-server assignment problem.

This is the bridge between our classical formulation (used by SA, GA, and
NSGA-II) and the quantum solver (QAOA, via Qiskit) planned next.

--------------------------------------------------------------------------
THE MATH
--------------------------------------------------------------------------
Decision variables:
    x[i][j] ∈ {0, 1}   for each task i = 1..N and server j = 1..M
    x[i][j] = 1  if task i is assigned to server j, else 0

Objective (minimize total energy cost):
    minimize  sum_i sum_j  c[i][j] * x[i][j]
    where     c[i][j] = size[i] * energy_cost_per_unit[j]

Constraints:
    (1) Each task is assigned to exactly one server:
            sum_j x[i][j] = 1        for every task i

    (2) Each server's total assigned load must not exceed its capacity:
            sum_i size[i] * x[i][j] <= capacity[j]     for every server j

--------------------------------------------------------------------------
FROM CONSTRAINED PROBLEM TO QUBO
--------------------------------------------------------------------------
QAOA (and QUBO solvers generally) can only minimize an UNCONSTRAINED
quadratic function of binary variables — there is no separate way to
express "must equal" or "must not exceed" constraints. So constraints are
converted into penalty terms added to the objective, each weighted by a
penalty strength (A, B) large enough that violating a constraint is always
more costly than any gain in the raw objective:

    H(x) = H_cost(x)  +  A * H_assignment(x)  +  B * H_capacity(x)

    H_cost(x)        = sum_i sum_j c[i][j] * x[i][j]
    H_assignment(x)   = sum_i ( 1 - sum_j x[i][j] )^2
    H_capacity(x)     = sum_j ( slack_j + sum_i size[i]*x[i][j] - capacity[j] )^2

H_assignment is zero exactly when every task is assigned to exactly one
server, and grows quadratically for every task assigned to zero or more
than one server. H_capacity uses an auxiliary "slack" variable (encoded in
binary) that can absorb any unused capacity, so the penalty is zero exactly
when the load is at or under capacity, and grows for any overflow.

This file uses Qiskit Optimization's QuadraticProgram to define the
constrained model exactly as above, then lets QuadraticProgramToQubo do the
penalty-conversion and slack-variable encoding automatically (rather than
hand-deriving the binary slack expansion), producing a ready-to-solve QUBO
and its equivalent Ising Hamiltonian for QAOA.

Usage:
    from data_generator import generate_scenario
    from qubo_formulation import build_qubo

    scenario = generate_scenario(num_tasks=10, num_servers=3, seed=42)
    qp, qubo, decode = build_qubo(scenario["tasks"], scenario["servers"])
"""

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo


def build_quadratic_program(tasks, servers, include_capacity=True):
    """
    Build the constrained QuadraticProgram (the "textbook" formulation,
    before any penalty conversion) for the task-to-server assignment problem.

    include_capacity: if False, the capacity constraint (2) is omitted,
        leaving only the assignment constraint (1). This matters because the
        capacity constraint requires binary-encoded slack variables whose
        qubit count grows with server capacity — on a local simulator this
        makes the full problem infeasible to run QAOA on beyond a handful of
        tasks (see qaoa_solver.py for the measured qubit-count/runtime
        scaling). Classical solvers (SA/GA/NSGA-II) always use the full,
        capacity-constrained problem; only the QAOA demo uses the reduced
        version, and this is documented explicitly as a NISQ-era scoping
        decision, not an omission.
    """
    qp = QuadraticProgram(name="QEDGE-OPT Task-to-Server Assignment")

    num_tasks = len(tasks)
    num_servers = len(servers)

    # --- Decision variables: x_i_j binary for every (task, server) pair ---
    var_name = lambda i, j: f"x_{i}_{j}"
    for i in range(num_tasks):
        for j in range(num_servers):
            qp.binary_var(name=var_name(i, j))

    # --- Objective: minimize total energy cost ---
    linear_terms = {}
    for i in range(num_tasks):
        for j in range(num_servers):
            cost_ij = tasks[i]["size"] * servers[j]["energy_cost_per_unit"]
            linear_terms[var_name(i, j)] = cost_ij
    qp.minimize(linear=linear_terms)

    # --- Constraint (1): each task assigned to exactly one server ---
    for i in range(num_tasks):
        coeffs = {var_name(i, j): 1 for j in range(num_servers)}
        qp.linear_constraint(linear=coeffs, sense="==", rhs=1, name=f"assign_task_{i}")

    # --- Constraint (2): each server's load must not exceed its capacity ---
    if include_capacity:
        for j in range(num_servers):
            coeffs = {var_name(i, j): tasks[i]["size"] for i in range(num_tasks)}
            qp.linear_constraint(
                linear=coeffs, sense="<=", rhs=servers[j]["capacity"], name=f"capacity_server_{j}"
            )

    return qp


def build_qubo(tasks, servers, penalty=None, include_capacity=True):
    """
    Build the constrained QuadraticProgram, then convert it to an
    unconstrained QUBO (penalty terms + slack variables added automatically).

    Returns:
        qp        : the original constrained QuadraticProgram (for reference)
        qubo      : the converted, unconstrained QUBO QuadraticProgram
        converter : the QuadraticProgramToQubo converter (used to decode a
                    QUBO solution back into the original x_i_j variables)
    """
    qp = build_quadratic_program(tasks, servers, include_capacity=include_capacity)
    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)
    return qp, qubo, converter


def decode_solution(x, tasks, servers, converter):
    """
    Given a raw QUBO bitstring solution `x` (as returned by a QUBO solver,
    e.g. QAOA), decode it back into a human-readable task -> server mapping.
    """
    # Map the QUBO solution back onto the original problem's variables
    original = converter.interpret(x)

    assignment = {}
    num_tasks = len(tasks)
    num_servers = len(servers)
    for i in range(num_tasks):
        for j in range(num_servers):
            var_index = i * num_servers + j
            if round(original[var_index]) == 1:
                assignment[tasks[i]["id"]] = servers[j]["id"]
    return assignment


if __name__ == "__main__":
    from data_generator import generate_scenario

    # QAOA/QUBO problems scale quickly (num_tasks * num_servers qubits, plus
    # slack qubits for the capacity constraint), so we use a SMALL toy
    # scenario here just to demonstrate the formulation works correctly —
    # this matches the "small problem instances" scope planned for QAOA.
    scenario = generate_scenario(num_tasks=4, num_servers=2, seed=42)
    tasks = scenario["tasks"]
    servers = scenario["servers"]

    print("=" * 60)
    print("QEDGE-OPT — QUBO Formulation Demo")
    print("=" * 60)

    print(f"\nToy scenario: {len(tasks)} tasks, {len(servers)} servers")
    for t in tasks:
        print(f"  {t['id']}: size={t['size']}, latency_requirement={t['latency_requirement']}")
    for s in servers:
        print(f"  {s['id']}: capacity={s['capacity']}, energy_cost_per_unit={s['energy_cost_per_unit']}")

    qp, qubo, converter = build_qubo(tasks, servers)

    print(f"\nConstrained model: {qp.get_num_binary_vars()} binary variables, "
          f"{len(qp.linear_constraints)} constraints")
    print(f"Converted QUBO: {qubo.get_num_binary_vars()} binary variables "
          f"(extra variables are binary-encoded slack for the capacity constraints)")

    print("\nQUBO is now ready to convert to an Ising Hamiltonian and solve with QAOA.")
    print("Next step: qubo.to_ising() -> run QAOA in Qiskit -> decode_solution() the result.")
