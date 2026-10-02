"""QEDGE-OPT comparison dashboard.

Run from the project folder:
    pip install streamlit pandas
    streamlit run app.py

Reads the CSV and PNG files already produced by run_benchmark.py,
exact_baseline.py, qaoa_control.py and demo.py. Missing files are skipped.
"""
from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).parent

st.set_page_config(page_title="QEDGE-OPT Dashboard", layout="wide")


@st.cache_data
def load_csv(name: str):
    path = BASE / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def need(df, name):
    if df is None:
        st.warning(f"`{name}` not found in the project folder. Run the script that creates it.")
        return False
    return True


def datasets_available(prefix_map):
    return [label for label, fname in prefix_map.items() if (BASE / fname).exists()]


# ---------------------------------------------------------------- header
st.title("QEDGE-OPT: Quantum-Assisted Edge Resource Allocation")
st.caption(
    "Benchmark of Simulated Annealing, Genetic Algorithm, NSGA-II, QAOA and an exact "
    "MILP baseline for task-to-server assignment on Alibaba trace and synthetic data."
)

tab_over, tab_gap, tab_qaoa, tab_ctrl, tab_charts, tab_raw = st.tabs(
    ["Overview", "Optimality gap", "QAOA", "QAOA control", "Charts", "Raw data"]
)

GAP_FILES = {
    "Alibaba": "exact_gap_alibaba_scipy.csv",
    "Synthetic": "exact_gap_synthetic_scipy.csv",
}
QAOA_FILES = {"Alibaba": "qaoa_alibaba.csv", "Synthetic": "qaoa_synthetic.csv"}
CTRL_FILES = {
    "Alibaba": "qaoa_control_alibaba.csv",
    "Synthetic": "qaoa_control_synthetic.csv",
}

# ---------------------------------------------------------------- overview
with tab_over:
    st.subheader("Solvers")
    st.markdown(
        """
- **Simulated Annealing (SA):** single-objective metaheuristic
- **Genetic Algorithm (GA):** single-objective metaheuristic
- **NSGA-II:** multi-objective (cost and latency violation)
- **QAOA:** quantum-assisted, assignment-only QUBO on a local Qiskit simulator, with hybrid repair
- **Exact (HiGHS MILP):** ground-truth optimum of the cost objective
        """
    )
    frames = []
    for label, fname in GAP_FILES.items():
        df = load_csv(fname)
        if df is not None:
            d = df.copy()
            d["dataset"] = label
            frames.append(d)
    if frames:
        allgap = pd.concat(frames, ignore_index=True)
        st.subheader("Mean optimality gap by method (all sizes and seeds)")
        summary = (
            allgap.groupby(["dataset", "method"])["gap_pct"].mean().round(2).unstack("method")
        )
        st.dataframe(summary, use_container_width=True)
        st.info(
            "The exact optimum ignores latency, so NSGA-II's gap partly reflects cost it "
            "trades for lower latency. It is not a pure search-quality gap."
        )
    else:
        st.warning("No optimality-gap CSVs found. Run `python exact_baseline.py`.")

# ---------------------------------------------------------------- gap
with tab_gap:
    st.subheader("Optimality gap vs. exact solver")
    avail = datasets_available(GAP_FILES)
    if not avail:
        st.warning("No `exact_gap_*_scipy.csv` files found. Run `python exact_baseline.py`.")
    else:
        c1, c2 = st.columns(2)
        ds = c1.selectbox("Dataset", avail, key="gap_ds")
        df = load_csv(GAP_FILES[ds])
        methods = sorted(df["method"].unique())
        chosen = c2.multiselect("Methods", methods, default=methods)
        view = df[df["method"].isin(chosen)]

        pivot = view.groupby(["num_tasks", "method"])["gap_pct"].mean().unstack("method").round(2)
        st.markdown("**Mean gap (%) over seeds, by number of tasks**")
        st.dataframe(pivot, use_container_width=True)
        st.line_chart(pivot)

        st.markdown("**Exact solver runtime (s)**")
        rt = (
            df.drop_duplicates(["num_tasks", "seed"])
            .groupby("num_tasks")["exact_runtime_s"]
            .mean()
            .round(3)
        )
        st.bar_chart(rt)

        with st.expander("Per-seed results"):
            st.dataframe(view, use_container_width=True)

# ---------------------------------------------------------------- qaoa
with tab_qaoa:
    st.subheader("QAOA results (assignment-only QUBO)")
    avail = datasets_available(QAOA_FILES)
    if not avail:
        st.warning("No `qaoa_*.csv` files found. Run `python run_benchmark.py`.")
    else:
        ds = st.selectbox("Dataset", avail, key="qaoa_ds")
        q = load_csv(QAOA_FILES[ds])
        st.dataframe(q, use_container_width=True)

        cols = st.columns(2)
        if {"num_tasks", "qaoa_runtime_s"}.issubset(q.columns):
            cols[0].markdown("**QAOA runtime (s) by problem size**")
            cols[0].bar_chart(q.set_index("num_tasks")["qaoa_runtime_s"])
        if {"num_tasks", "qaoa_capacity_overflow"}.issubset(q.columns):
            cols[1].markdown("**Capacity overflow before repair**")
            cols[1].bar_chart(q.set_index("num_tasks")["qaoa_capacity_overflow"])
        if {"num_tasks", "gap_repaired_pct"}.issubset(q.columns):
            st.markdown("**Gap to optimum after hybrid repair (%)**")
            st.bar_chart(q.set_index("num_tasks")["gap_repaired_pct"])

        st.info(
            "The QUBO encodes assignment only, so QAOA can overload servers. The large "
            "unrepaired real-world gaps come from capacity overflow, not from QAOA failing. "
            "Qubit count limits QAOA to very small instances."
        )

# ---------------------------------------------------------------- control
with tab_ctrl:
    st.subheader("QAOA control experiment")
    avail = datasets_available(CTRL_FILES)
    if not avail:
        st.warning("No `qaoa_control_*.csv` files found. Run `python qaoa_control.py`.")
    else:
        ds = st.selectbox("Dataset", avail, key="ctrl_ds")
        c = load_csv(CTRL_FILES[ds])

        hit = c.groupby("method")["hit_optimum"].apply(lambda s: (s == 1).sum())
        total = c.groupby("method")["hit_optimum"].count()
        out = pd.DataFrame({"runs_reaching_optimum": hit, "total_runs": total})
        st.markdown("**Runs where the method reached the true optimum after repair**")
        st.dataframe(out, use_container_width=True)
        st.bar_chart(out["runs_reaching_optimum"])

        st.markdown("**Mean repaired gap (%) by size and method**")
        gap = (
            c.groupby(["num_tasks", "method"])["repaired_gap_pct"].mean().unstack("method").round(2)
        )
        st.dataframe(gap, use_container_width=True)

        with st.expander("Per-seed results"):
            st.dataframe(c, use_container_width=True)

# ---------------------------------------------------------------- charts
with tab_charts:
    st.subheader("Saved charts")
    images = [
        ("benchmark_cost.png", "Benchmark cost"),
        ("benchmark_runtime.png", "Benchmark runtime"),
        ("qaoa_scaling.png", "QAOA scaling"),
        ("method_comparison.png", "SA / GA / NSGA-II cost"),
        ("qaoa_comparison.png", "QAOA vs. exact"),
        ("nsga2_pareto.png", "NSGA-II Pareto front"),
        ("annealing_progress.png", "SA progress"),
        ("ga_progress.png", "GA progress"),
        ("nsga2_progress.png", "NSGA-II progress"),
    ]
    shown = 0
    cols = st.columns(2)
    for fname, caption in images:
        p = BASE / fname
        if p.exists():
            cols[shown % 2].image(str(p), caption=caption, use_container_width=True)
            shown += 1
    if shown == 0:
        st.warning("No chart PNGs found. Run `python demo.py` and `python run_benchmark.py`.")

# ---------------------------------------------------------------- raw
with tab_raw:
    st.subheader("Raw result files")
    csvs = sorted(p.name for p in BASE.glob("*.csv"))
    if not csvs:
        st.warning("No CSV files found in the project folder.")
    else:
        pick = st.selectbox("File", csvs)
        df = load_csv(pick)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=pick)
