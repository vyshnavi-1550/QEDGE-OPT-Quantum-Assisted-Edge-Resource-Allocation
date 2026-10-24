
"""QEDGE-OPT comparison dashboard.

Run from the project folder:
    python -m pip install streamlit pandas matplotlib
    python -m streamlit run app.py

Uses the latest benchmark outputs where available and falls back to
older dataset-specific files only when the latest file is unavailable.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="QEDGE-OPT Dashboard",
    page_icon="⚛️",
    layout="wide",
)

# ---------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_csv(name: str):
    """Load a CSV file from the project folder."""
    path = BASE / name

    if not path.is_file():
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read {name}: {exc}")
        return None


def has_columns(df, columns):
    """Return True when a DataFrame contains all required columns."""
    return df is not None and set(columns).issubset(df.columns)


def available_files(file_map):
    """Return display labels for files that actually exist."""
    return [
        label
        for label, filename in file_map.items()
        if (BASE / filename).is_file()
    ]


def show_missing(message):
    st.warning(message)


def numeric_column(df, column):
    """Convert a column to numeric values where possible."""
    return pd.to_numeric(df[column], errors="coerce")


# ---------------------------------------------------------------------
# File definitions
# ---------------------------------------------------------------------

GAP_FILES = {
    "Alibaba": "exact_gap_alibaba_scipy.csv",
    "Synthetic": "exact_gap_synthetic_scipy.csv",
}

# Prefer the latest combined QAOA benchmark.
# Older files remain available as fallbacks.
QAOA_FILES = {
    "Latest benchmark": "qaoa_results.csv",
    "Alibaba (older benchmark)": "qaoa_alibaba.csv",
    "Synthetic (older benchmark)": "qaoa_synthetic.csv",
}

CTRL_FILES = {
    "Alibaba": "qaoa_control_alibaba.csv",
    "Synthetic": "qaoa_control_synthetic.csv",
}

CHART_FILES = [
    ("benchmark_cost.png", "Benchmark cost"),
    ("benchmark_runtime.png", "Benchmark runtime"),
    ("qaoa_scaling.png", "QAOA scaling"),
    ("method_comparison.png", "Method comparison"),
    ("qaoa_comparison.png", "QAOA vs. exact"),
    ("nsga2_pareto.png", "NSGA-II Pareto front"),
    ("annealing_progress.png", "Simulated Annealing progress"),
    ("ga_progress.png", "Genetic Algorithm progress"),
    ("nsga2_progress.png", "NSGA-II progress"),
]


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

st.title("QEDGE-OPT: Quantum-Assisted Edge Resource Allocation")

st.caption(
    "Benchmark of Simulated Annealing, Genetic Algorithm, NSGA-II, "
    "QAOA and an exact MILP baseline for task-to-server assignment "
    "on Alibaba trace-derived and synthetic data."
)

tab_over, tab_gap, tab_qaoa, tab_ctrl, tab_charts, tab_raw = st.tabs(
    [
        "Overview",
        "Optimality gap",
        "QAOA",
        "QAOA control",
        "Charts",
        "Raw data",
    ]
)


# ---------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------

with tab_over:
    st.subheader("Solvers")

    st.markdown(
        """
- **Simulated Annealing (SA):** single-objective classical metaheuristic.
- **Genetic Algorithm (GA):** evolutionary search over assignments.
- **NSGA-II:** multi-objective optimization of cost and latency violation.
- **QAOA:** quantum-assisted optimization of an assignment-only QUBO,
  followed by classical capacity repair.
- **Exact baseline:** SciPy/HiGHS MILP optimization for comparison
  with heuristic solutions.
        """
    )

    frames = []

    for dataset, filename in GAP_FILES.items():
        df = load_csv(filename)

        if df is not None and has_columns(
            df, ["method", "gap_pct"]
        ):
            copy = df.copy()
            copy["dataset"] = dataset
            copy["gap_pct"] = numeric_column(copy, "gap_pct")
            frames.append(copy)

    if frames:
        all_gaps = pd.concat(frames, ignore_index=True)

        st.subheader("Mean optimality gap by method")

        summary = (
            all_gaps.dropna(subset=["gap_pct"])
            .groupby(["dataset", "method"])["gap_pct"]
            .mean()
            .round(2)
            .unstack("method")
        )

        st.dataframe(summary, use_container_width=True)

        st.info(
            "Interpret optimality gaps only when the heuristic and exact "
            "baseline use compatible objectives and constraints. If NSGA-II "
            "trades cost for lower latency, its cost gap alone does not "
            "measure search quality."
        )
    else:
        show_missing(
            "No usable exact-gap CSV files were found. "
            "Run `python exact_baseline.py`."
        )

    st.subheader("Latest classical benchmark")

    benchmark = load_csv("benchmark_results.csv")

    if benchmark is not None:
        st.dataframe(benchmark, use_container_width=True)
    else:
        show_missing(
            "benchmark_results.csv not found. "
            "Run `python run_benchmark.py`."
        )


# ---------------------------------------------------------------------
# Optimality gap
# ---------------------------------------------------------------------

with tab_gap:
    st.subheader("Optimality gap vs. exact baseline")

    datasets = available_files(GAP_FILES)

    if not datasets:
        show_missing(
            "No exact-gap CSV files found. "
            "Run `python exact_baseline.py`."
        )
    else:
        col1, col2 = st.columns(2)

        dataset = col1.selectbox(
            "Dataset",
            datasets,
            key="gap_dataset",
        )

        df = load_csv(GAP_FILES[dataset])

        if not has_columns(
            df, ["num_tasks", "seed", "method", "gap_pct"]
        ):
            show_missing(
                "The selected CSV is missing required columns: "
                "num_tasks, seed, method, gap_pct."
            )
        else:
            methods = sorted(df["method"].dropna().unique().tolist())

            selected_methods = col2.multiselect(
                "Methods",
                methods,
                default=methods,
                key="gap_methods",
            )

            view = df[df["method"].isin(selected_methods)].copy()
            view["gap_pct"] = numeric_column(view, "gap_pct")
            view["num_tasks"] = numeric_column(view, "num_tasks")

            pivot = (
                view.dropna(subset=["num_tasks", "gap_pct"])
                .groupby(["num_tasks", "method"])["gap_pct"]
                .mean()
                .unstack("method")
                .sort_index()
                .round(2)
            )

            st.markdown("**Mean gap (%) by problem size**")
            st.dataframe(pivot, use_container_width=True)

            if not pivot.empty:
                st.line_chart(pivot)

            if "exact_runtime_s" in df.columns:
                st.markdown("**Exact solver runtime (seconds)**")

                runtime_df = df.copy()
                runtime_df["exact_runtime_s"] = numeric_column(
                    runtime_df, "exact_runtime_s"
                )
                runtime_df["num_tasks"] = numeric_column(
                    runtime_df, "num_tasks"
                )

                runtime = (
                    runtime_df.dropna(
                        subset=["num_tasks", "seed", "exact_runtime_s"]
                    )
                    .drop_duplicates(["num_tasks", "seed"])
                    .groupby("num_tasks")["exact_runtime_s"]
                    .mean()
                    .sort_index()
                    .round(3)
                )

                if not runtime.empty:
                    st.bar_chart(runtime)

            with st.expander("Per-seed results"):
                st.dataframe(view, use_container_width=True)


# ---------------------------------------------------------------------
# QAOA results
# ---------------------------------------------------------------------

with tab_qaoa:
    st.subheader("QAOA results")

    datasets = available_files(QAOA_FILES)

    if not datasets:
        show_missing(
            "No QAOA result files found. "
            "Run `python run_benchmark.py`."
        )
    else:
        # The latest combined benchmark is selected by default.
        dataset = st.selectbox(
            "Dataset / result file",
            datasets,
            index=0,
            key="qaoa_dataset",
        )

        qaoa = load_csv(QAOA_FILES[dataset])

        if qaoa is None or qaoa.empty:
            show_missing("The selected QAOA CSV is empty or unavailable.")
        else:
            st.dataframe(qaoa, use_container_width=True)

            # Plot only metrics that are present in the selected file.
            if has_columns(qaoa, ["num_tasks", "qaoa_runtime_s"]):
                st.markdown("**QAOA runtime by problem size (seconds)**")

                runtime_df = qaoa[
                    ["num_tasks", "qaoa_runtime_s"]
                ].copy()

                runtime_df["num_tasks"] = numeric_column(
                    runtime_df, "num_tasks"
                )
                runtime_df["qaoa_runtime_s"] = numeric_column(
                    runtime_df, "qaoa_runtime_s"
                )

                runtime_series = (
                    runtime_df.dropna()
                    .groupby("num_tasks")["qaoa_runtime_s"]
                    .mean()
                    .sort_index()
                )

                if not runtime_series.empty:
                    st.bar_chart(runtime_series)

            if has_columns(
                qaoa, ["num_tasks", "qaoa_capacity_overflow"]
            ):
                st.markdown("**Capacity overflow before repair**")

                overflow_df = qaoa[
                    ["num_tasks", "qaoa_capacity_overflow"]
                ].copy()

                overflow_df["num_tasks"] = numeric_column(
                    overflow_df, "num_tasks"
                )
                overflow_df["qaoa_capacity_overflow"] = numeric_column(
                    overflow_df, "qaoa_capacity_overflow"
                )

                overflow = (
                    overflow_df.dropna()
                    .groupby("num_tasks")["qaoa_capacity_overflow"]
                    .mean()
                    .sort_index()
                )

                if not overflow.empty:
                    st.bar_chart(overflow)

            if has_columns(
                qaoa, ["num_tasks", "gap_repaired_pct"]
            ):
                st.markdown("**Gap after classical capacity repair (%)**")

                repaired_df = qaoa[
                    ["num_tasks", "gap_repaired_pct"]
                ].copy()

                repaired_df["num_tasks"] = numeric_column(
                    repaired_df, "num_tasks"
                )
                repaired_df["gap_repaired_pct"] = numeric_column(
                    repaired_df, "gap_repaired_pct"
                )

                repaired_gap = (
                    repaired_df.dropna()
                    .groupby("num_tasks")["gap_repaired_pct"]
                    .mean()
                    .sort_index()
                )

                if not repaired_gap.empty:
                    st.bar_chart(repaired_gap)

            st.info(
                "The current QAOA workflow uses an assignment-only QUBO "
                "by default, then applies classical capacity repair. "
                "Therefore, it is a hybrid method and does not directly "
                "solve the full capacity-constrained problem. The reported "
                "small-instance results do not establish quantum advantage."
            )


# ---------------------------------------------------------------------
# QAOA control experiment
# ---------------------------------------------------------------------

with tab_ctrl:
    st.subheader("QAOA control experiment")

    datasets = available_files(CTRL_FILES)

    if not datasets:
        show_missing(
            "No QAOA control CSV files found. "
            "Run `python qaoa_control.py`."
        )
    else:
        dataset = st.selectbox(
            "Dataset",
            datasets,
            key="control_dataset",
        )

        control = load_csv(CTRL_FILES[dataset])

        if control is None or control.empty:
            show_missing("The selected control CSV is empty.")
        elif not has_columns(
            control, ["method", "hit_optimum"]
        ):
            show_missing(
                "The control CSV must contain method and hit_optimum columns."
            )
        else:
            control = control.copy()
            control["hit_optimum"] = numeric_column(
                control, "hit_optimum"
            )

            # Binary indicators can be counted as successful runs.
            # Fractional/averaged values must instead be summarized as means.
            st.markdown("**Optimality success summary**")

            summary_rows = []

            for method, group in control.groupby("method"):
                values = group["hit_optimum"].dropna()

                if values.empty:
                    continue

                is_binary = values.isin([0, 1]).all()

                if is_binary:
                    summary_rows.append(
                        {
                            "method": method,
                            "metric": "Successful runs",
                            "value": int(values.sum()),
                            "total_rows": int(values.count()),
                            "success_rate_pct": round(
                                values.mean() * 100, 2
                            ),
                        }
                    )
                else:
                    summary_rows.append(
                        {
                            "method": method,
                            "metric": "Mean recorded hit_optimum",
                            "value": round(float(values.mean()), 4),
                            "total_rows": int(values.count()),
                            "success_rate_pct": None,
                        }
                    )

            if summary_rows:
                summary = pd.DataFrame(summary_rows)
                st.dataframe(summary, use_container_width=True)

                # Plot the mean recorded metric; do not present fractional
                # random-baseline averages as discrete successful runs.
                means = (
                    control.groupby("method")["hit_optimum"]
                    .mean()
                    .sort_values(ascending=False)
                )

                st.markdown("**Mean recorded hit_optimum by method**")
                st.bar_chart(means)

            if has_columns(
                control, ["num_tasks", "method", "repaired_gap_pct"]
            ):
                st.markdown("**Mean repaired gap (%) by problem size**")

                gap_df = control.copy()
                gap_df["num_tasks"] = numeric_column(
                    gap_df, "num_tasks"
                )
                gap_df["repaired_gap_pct"] = numeric_column(
                    gap_df, "repaired_gap_pct"
                )

                gap = (
                    gap_df.dropna(
                        subset=["num_tasks", "repaired_gap_pct"]
                    )
                    .groupby(["num_tasks", "method"])[
                        "repaired_gap_pct"
                    ]
                    .mean()
                    .unstack("method")
                    .sort_index()
                    .round(2)
                )

                st.dataframe(gap, use_container_width=True)

                if not gap.empty:
                    st.line_chart(gap)

            with st.expander("Per-seed / raw control results"):
                st.dataframe(control, use_container_width=True)


# ---------------------------------------------------------------------
# Saved charts
# ---------------------------------------------------------------------

with tab_charts:
    st.subheader("Saved charts")

    shown = 0
    columns = st.columns(2)

    for filename, caption in CHART_FILES:
        path = BASE / filename

        if path.is_file():
            with columns[shown % 2]:
                st.image(
                    str(path),
                    caption=caption,
                    use_container_width=True,
                )
            shown += 1

    if shown == 0:
        show_missing(
            "No chart PNGs found. Run `python demo.py` and "
            "`python run_benchmark.py`."
        )


# ---------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------

with tab_raw:
    st.subheader("Raw result files")

    csv_files = sorted(path.name for path in BASE.glob("*.csv"))

    if not csv_files:
        show_missing("No CSV files found in the project folder.")
    else:
        selected_file = st.selectbox(
            "CSV file",
            csv_files,
            key="raw_csv_file",
        )

        raw_df = load_csv(selected_file)

        if raw_df is not None:
            st.caption(
                f"File: {selected_file} · "
                f"{len(raw_df)} rows · {len(raw_df.columns)} columns"
            )

            st.dataframe(raw_df, use_container_width=True)

            st.download_button(
                label="Download CSV",
                data=raw_df.to_csv(index=False).encode("utf-8"),
                file_name=selected_file,
                mime="text/csv",
            )
