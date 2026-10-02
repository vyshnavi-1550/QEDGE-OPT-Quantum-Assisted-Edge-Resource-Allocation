"""QEDGE-OPT advanced dashboard.

Run from the project folder:
    python -m pip install streamlit pandas
    python -m streamlit run app_advanced.py

Reads the CSV / PNG files produced by run_benchmark.py, exact_baseline.py,
qaoa_control.py and demo.py. Missing files are skipped with a warning.
Charts use Altair, which is installed together with Streamlit.
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
TOL = 0.005  # gap differences below this (in %) count as a tie

st.set_page_config(page_title="QEDGE-OPT Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 10px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

GAP_FILES = ["exact_gap_alibaba_scipy.csv", "exact_gap_synthetic_scipy.csv"]
QAOA_FILES = ["qaoa_alibaba.csv", "qaoa_synthetic.csv"]
CTRL_FILES = ["qaoa_control_alibaba.csv", "qaoa_control_synthetic.csv"]


# ------------------------------------------------------------------ data
@st.cache_data
def load_many(files):
    frames = []
    for f in files:
        p = BASE / f
        if p.exists():
            d = pd.read_csv(p)
            if "data" in d.columns:
                d["dataset"] = d["data"].astype(str).str.capitalize()
            else:
                d["dataset"] = f
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_one(name):
    p = BASE / name
    return pd.read_csv(p) if p.exists() else None


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


gap = load_many(GAP_FILES)
qaoa = load_many(QAOA_FILES)
ctrl = load_many(CTRL_FILES)

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Filters")
if st.sidebar.button("Reload data files"):
    st.cache_data.clear()
    st.rerun()

if gap.empty:
    st.title("QEDGE-OPT Dashboard")
    st.error("No `exact_gap_*_scipy.csv` files found. Run `python exact_baseline.py` first.")
    st.stop()

all_ds = sorted(gap["dataset"].unique())
all_methods = sorted(gap["method"].unique())
all_sizes = sorted(gap["num_tasks"].unique())

sel_ds = st.sidebar.multiselect("Dataset", all_ds, default=all_ds)
sel_methods = st.sidebar.multiselect("Methods", all_methods, default=all_methods)
sel_sizes = st.sidebar.multiselect("Problem sizes (tasks)", all_sizes, default=all_sizes)
st.sidebar.caption(
    "Filters apply to the gap analysis tabs. QAOA tabs show their own instances."
)

gf = gap[
    gap["dataset"].isin(sel_ds)
    & gap["method"].isin(sel_methods)
    & gap["num_tasks"].isin(sel_sizes)
].copy()

# ------------------------------------------------------------------ header
st.title("QEDGE-OPT: Quantum-Assisted Edge Resource Allocation")
st.caption(
    "Benchmark of SA, GA, NSGA-II and QAOA against an exact MILP baseline "
    "on Alibaba trace and synthetic workloads."
)

if gf.empty:
    st.warning("The current filters match no data. Widen the selection in the sidebar.")
    st.stop()

mean_by_method = gf.groupby("method")["gap_pct"].mean().sort_values()
best_method = mean_by_method.index[0]
rt = gf.drop_duplicates(["dataset", "num_tasks", "seed"])
largest = gf["num_tasks"].max()
rt_largest = rt[rt["num_tasks"] == largest]["exact_runtime_s"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Best method (lowest mean gap)", best_method, f"{mean_by_method.iloc[0]:.2f}% gap")
k2.metric("Worst method (highest mean gap)", mean_by_method.index[-1], f"{mean_by_method.iloc[-1]:.2f}% gap")
k3.metric(f"Exact solver time at {largest} tasks", f"{rt_largest:.2f} s")
if not ctrl.empty:
    q = ctrl[ctrl["method"] == "QAOA"]
    hits = int((q["hit_optimum"] == 1).sum())
    k4.metric("QAOA runs reaching optimum", f"{hits} / {len(q)}")
else:
    k4.metric("QAOA runs reaching optimum", "n/a")

tabs = st.tabs(
    [
        "Summary",
        "Gap explorer",
        "Head-to-head",
        "Exact solver",
        "QAOA",
        "QAOA control",
        "Charts",
        "Data & report",
    ]
)

# ------------------------------------------------------------------ summary
with tabs[0]:
    st.subheader("Mean optimality gap (%) by method and size")
    agg = (
        gf.groupby(["dataset", "num_tasks", "method"])["gap_pct"]
        .agg(mean="mean", std="std", runs="count")
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0)
    agg["mean"] = agg["mean"].round(2)

    cols = st.columns(len(sel_ds)) if sel_ds else []
    for col, ds in zip(cols, sel_ds):
        d = agg[agg["dataset"] == ds]
        base = alt.Chart(d).encode(
            x=alt.X("num_tasks:O", title="Tasks"),
            y=alt.Y("method:N", title=None),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("mean:Q", scale=alt.Scale(scheme="orangered"), title="Gap %"),
            tooltip=["dataset", "num_tasks", "method", "mean", "std", "runs"],
        )
        text = base.mark_text(fontSize=12).encode(text=alt.Text("mean:Q", format=".2f"))
        col.markdown(f"**{ds}**")
        col.altair_chart((heat + text).properties(height=160), use_container_width=True)

    st.subheader("Key findings")
    lines = []
    for ds in sel_ds:
        d = gf[gf["dataset"] == ds]
        m = d.groupby("method")["gap_pct"].mean().sort_values()
        lines.append(
            f"- **{ds}:** {m.index[0]} has the lowest mean gap ({m.iloc[0]:.2f}%), "
            f"{m.index[-1]} the highest ({m.iloc[-1]:.2f}%)."
        )
        top = d[d["num_tasks"] == d["num_tasks"].max()].groupby("method")["gap_pct"].mean()
        parts = ", ".join(f"{k} {v:.2f}%" for k, v in top.sort_values().items())
        lines.append(f"  - At {int(d['num_tasks'].max())} tasks: {parts}.")
    lines.append(
        f"- The exact MILP solver needs on average {rt_largest:.2f} s at {largest} tasks, "
        "so it is a strong baseline at these sizes."
    )
    lines.append(
        "- The exact optimum ignores latency, so NSGA-II's gap partly reflects cost it "
        "trades for lower latency."
    )
    st.markdown("\n".join(lines))

# ------------------------------------------------------------------ gap explorer
with tabs[1]:
    st.subheader("Gap vs. problem size (mean with ±1 std over seeds)")
    cols = st.columns(len(sel_ds)) if sel_ds else []
    for col, ds in zip(cols, sel_ds):
        d = gf[gf["dataset"] == ds]
        b = alt.Chart(d)
        err = b.mark_errorbar(extent="stdev", ticks=True).encode(
            x=alt.X("num_tasks:O", title="Tasks"),
            y=alt.Y("gap_pct:Q", title="Gap (%)"),
            color="method:N",
        )
        line = b.mark_line(point=True).encode(
            x=alt.X("num_tasks:O", title="Tasks"),
            y=alt.Y("mean(gap_pct):Q", title="Gap (%)"),
            color="method:N",
            tooltip=["method", "num_tasks", alt.Tooltip("mean(gap_pct):Q", format=".2f")],
        )
        col.markdown(f"**{ds}**")
        col.altair_chart((err + line).properties(height=300), use_container_width=True)

    st.subheader("Spread across seeds")
    box = (
        alt.Chart(gf)
        .mark_boxplot()
        .encode(
            x=alt.X("method:N", title=None),
            y=alt.Y("gap_pct:Q", title="Gap (%)"),
            color="method:N",
            column=alt.Column("num_tasks:O", title="Tasks"),
        )
        .properties(width=90, height=220)
    )
    st.altair_chart(box)

    st.subheader("Solution cost vs. exact optimum (each point is one run)")
    sc = (
        alt.Chart(gf)
        .mark_circle(size=70, opacity=0.75)
        .encode(
            x=alt.X("exact_cost:Q", title="Exact optimal cost", scale=alt.Scale(type="log")),
            y=alt.Y("cost:Q", title="Method cost", scale=alt.Scale(type="log")),
            color="method:N",
            shape="dataset:N",
            tooltip=["dataset", "num_tasks", "seed", "method", "cost", "exact_cost", "gap_pct"],
        )
        .properties(height=340)
    )
    diag = (
        alt.Chart(pd.DataFrame({"v": [gf["exact_cost"].min(), gf["exact_cost"].max()]}))
        .mark_line(color="gray", strokeDash=[4, 4])
        .encode(x="v:Q", y="v:Q")
    )
    st.altair_chart(sc + diag, use_container_width=True)
    st.caption("Points on the dashed line equal the exact optimum. Points above it cost more.")

# ------------------------------------------------------------------ head to head
with tabs[2]:
    st.subheader("Head-to-head comparison")
    if len(all_methods) < 2:
        st.info("Need at least two methods.")
    else:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Method A", all_methods, index=0)
        b_opts = [m for m in all_methods if m != a]
        b = c2.selectbox("Method B", b_opts, index=0)

        h = gap[
            gap["dataset"].isin(sel_ds)
            & gap["num_tasks"].isin(sel_sizes)
            & gap["method"].isin([a, b])
        ]
        pv = h.pivot_table(
            index=["dataset", "num_tasks", "seed"], columns="method", values="gap_pct"
        ).reset_index()
        if a in pv.columns and b in pv.columns and not pv.empty:
            pv["diff"] = pv[a] - pv[b]
            wins_a = int((pv["diff"] < -TOL).sum())
            wins_b = int((pv["diff"] > TOL).sum())
            ties = int((pv["diff"].abs() <= TOL).sum())
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{a} better (lower gap)", wins_a)
            m2.metric("Tie", ties)
            m3.metric(f"{b} better (lower gap)", wins_b)

            sc2 = (
                alt.Chart(pv)
                .mark_circle(size=80, opacity=0.8)
                .encode(
                    x=alt.X(f"{a}:Q", title=f"{a} gap (%)"),
                    y=alt.Y(f"{b}:Q", title=f"{b} gap (%)"),
                    color="num_tasks:O",
                    shape="dataset:N",
                    tooltip=["dataset", "num_tasks", "seed", a, b, "diff"],
                )
                .properties(height=340)
            )
            mx = float(max(pv[a].max(), pv[b].max(), 0.1))
            ln = (
                alt.Chart(pd.DataFrame({"v": [0, mx]}))
                .mark_line(color="gray", strokeDash=[4, 4])
                .encode(x="v:Q", y="v:Q")
            )
            st.altair_chart(sc2 + ln, use_container_width=True)
            st.caption(
                f"Points below the dashed line: {b} has the lower gap. "
                f"Points above: {a} has the lower gap."
            )
            st.dataframe(pv.round(3), use_container_width=True)
        else:
            st.info("No paired runs for the current selection.")

# ------------------------------------------------------------------ exact solver
with tabs[3]:
    st.subheader("Exact solver (HiGHS MILP) runtime")
    ert = (
        gap[gap["dataset"].isin(sel_ds)]
        .drop_duplicates(["dataset", "num_tasks", "seed"])
        .groupby(["dataset", "num_tasks"])["exact_runtime_s"]
        .mean()
        .reset_index()
    )
    ch = (
        alt.Chart(ert)
        .mark_line(point=True)
        .encode(
            x=alt.X("num_tasks:O", title="Tasks"),
            y=alt.Y("exact_runtime_s:Q", title="Mean runtime (s, log)", scale=alt.Scale(type="log")),
            color="dataset:N",
            tooltip=["dataset", "num_tasks", alt.Tooltip("exact_runtime_s:Q", format=".3f")],
        )
        .properties(height=320)
    )
    st.altair_chart(ch, use_container_width=True)
    if "exact_status" in gap.columns:
        st.write("Solver status counts:", gap.drop_duplicates(["dataset", "num_tasks", "seed"])["exact_status"].value_counts().to_dict())
    st.info(
        "At these scales the exact solver proves optimality in about a second or two, "
        "so metaheuristics are justified mainly for the multi-objective version and for "
        "larger or non-linear extensions."
    )

# ------------------------------------------------------------------ QAOA
with tabs[4]:
    st.subheader("QAOA (assignment-only QUBO, local simulator)")
    if qaoa.empty:
        st.warning("No `qaoa_*.csv` files found. Run `python run_benchmark.py`.")
    else:
        qd = st.multiselect(
            "Dataset", sorted(qaoa["dataset"].unique()), default=sorted(qaoa["dataset"].unique()), key="qd"
        )
        q = qaoa[qaoa["dataset"].isin(qd)]
        st.dataframe(q.drop(columns=["dataset"], errors="ignore"), use_container_width=True)

        c1, c2 = st.columns(2)
        if {"qubits", "qaoa_runtime_s"}.issubset(q.columns):
            rtc = (
                alt.Chart(q)
                .mark_line(point=True)
                .encode(
                    x=alt.X("qubits:O", title="Qubits"),
                    y=alt.Y("qaoa_runtime_s:Q", title="QAOA runtime (s, log)", scale=alt.Scale(type="log")),
                    color="dataset:N",
                    tooltip=["dataset", "num_tasks", "qubits", "qaoa_runtime_s"],
                )
                .properties(height=280, title="Simulation time grows with qubit count")
            )
            c1.altair_chart(rtc, use_container_width=True)
        if {"num_tasks", "gap_real_world_pct", "gap_repaired_pct"}.issubset(q.columns):
            m = q.melt(
                id_vars=["dataset", "num_tasks"],
                value_vars=["gap_real_world_pct", "gap_repaired_pct"],
                var_name="stage",
                value_name="gap",
            )
            m["stage"] = m["stage"].map(
                {"gap_real_world_pct": "Raw (with overflow)", "gap_repaired_pct": "After repair"}
            )
            gc = (
                alt.Chart(m)
                .mark_bar()
                .encode(
                    x=alt.X("stage:N", title=None),
                    y=alt.Y("gap:Q", title="Gap (%, symlog)", scale=alt.Scale(type="symlog")),
                    color="stage:N",
                    column=alt.Column("num_tasks:O", title="Tasks"),
                    tooltip=["dataset", "num_tasks", "stage", "gap"],
                )
                .properties(width=70, height=260, title="Gap before and after repair")
            )
            c2.altair_chart(gc)
        if {"num_tasks", "qaoa_capacity_overflow"}.issubset(q.columns):
            st.markdown("**Capacity overflow of raw QAOA solutions**")
            st.bar_chart(q.set_index("num_tasks")["qaoa_capacity_overflow"])

        st.info(
            "The QUBO encodes assignment only, so QAOA can overload servers. The very large raw "
            "gaps come from that missing constraint, not from the optimizer failing. "
            "Hybrid repair restores feasibility."
        )

# ------------------------------------------------------------------ QAOA control
with tabs[5]:
    st.subheader("Does QAOA beat simple baselines?")
    if ctrl.empty:
        st.warning("No `qaoa_control_*.csv` files found. Run `python qaoa_control.py`.")
    else:
        cd = st.multiselect(
            "Dataset", sorted(ctrl["dataset"].unique()), default=sorted(ctrl["dataset"].unique()), key="cd"
        )
        c = ctrl[ctrl["dataset"].isin(cd)].copy()
        c["reached"] = (c["hit_optimum"] == 1).astype(int)

        tot = c.groupby("method")["reached"].agg(reached="sum", runs="count").reset_index()
        tot["rate_%"] = (100 * tot["reached"] / tot["runs"]).round(1)
        st.dataframe(tot, use_container_width=True)

        bysize = c.groupby(["method", "num_tasks"])["reached"].sum().reset_index()
        bc = (
            alt.Chart(bysize)
            .mark_bar()
            .encode(
                x=alt.X("method:N", title=None),
                y=alt.Y("reached:Q", title="Runs reaching optimum"),
                color="method:N",
                column=alt.Column("num_tasks:O", title="Tasks"),
                tooltip=["method", "num_tasks", "reached"],
            )
            .properties(width=110, height=240)
        )
        st.altair_chart(bc)

        pv = c.pivot_table(
            index=["dataset", "num_tasks", "seed"], columns="method", values="repaired_gap_pct"
        ).reset_index()
        if {"QAOA", "Exact-QUBO"}.issubset(pv.columns):
            d = pv["QAOA"] - pv["Exact-QUBO"]
            m1, m2, m3 = st.columns(3)
            m1.metric("QAOA better than Exact-QUBO", int((d < -TOL).sum()))
            m2.metric("Tie", int((d.abs() <= TOL).sum()))
            m3.metric("Exact-QUBO better", int((d > TOL).sum()))
        with st.expander("Per-instance repaired gap (%)"):
            st.dataframe(pv.round(2), use_container_width=True)

        st.info(
            "Where QAOA misses the optimum, the exact solution of the same QUBO misses by the "
            "same amount, so the shortfall comes from the assignment-only formulation. "
            "A cheapest-server rule performs on par with QAOA."
        )

# ------------------------------------------------------------------ charts
with tabs[6]:
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
    found = [(f, c) for f, c in images if (BASE / f).exists()]
    if not found:
        st.warning("No chart PNGs found. Run `python demo.py` and `python run_benchmark.py`.")
    else:
        cols = st.columns(2)
        for i, (f, c) in enumerate(found):
            cols[i % 2].image(str(BASE / f), caption=c, use_container_width=True)

# ------------------------------------------------------------------ data & report
with tabs[7]:
    st.subheader("Summary report")
    summ = (
        gf.groupby(["dataset", "num_tasks", "method"])["gap_pct"].mean().round(2).unstack("method").reset_index()
    )
    report = [
        "# QEDGE-OPT benchmark summary",
        "",
        f"Datasets: {', '.join(sel_ds)}. Methods: {', '.join(sel_methods)}. Sizes: {', '.join(map(str, sel_sizes))}.",
        "",
        "## Mean optimality gap (%)",
        md_table(summ),
        "",
        f"Exact solver mean runtime at {largest} tasks: {rt_largest:.2f} s.",
    ]
    if not ctrl.empty:
        cc = ctrl.copy()
        cc["reached"] = (cc["hit_optimum"] == 1).astype(int)
        t = cc.groupby("method")["reached"].agg(reached="sum", runs="count").reset_index()
        report += ["", "## QAOA control (runs reaching the optimum)", md_table(t)]
    report_text = "\n".join(report)
    st.download_button("Download report (.md)", report_text, file_name="qedge_opt_summary.md")
    with st.expander("Preview"):
        st.markdown(report_text)

    st.subheader("Raw result files")
    csvs = sorted(p.name for p in BASE.glob("*.csv"))
    if csvs:
        pick = st.selectbox("File", csvs)
        df = load_one(pick)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=pick)
    else:
        st.warning("No CSV files found in the project folder.")
