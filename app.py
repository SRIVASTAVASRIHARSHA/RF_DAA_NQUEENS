"""Streamlit dashboard for N-Queens benchmark analysis and solver replay."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.iterative_solver import solve_n_queens_iterative
from src.ml_guided_solver import load_trained_model, solve_n_queens_ml
from src.recursive_solver import solve_n_queens
from src.reporting import generate_report

ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "results" / "benchmark"
SUMMARY_PATH = BENCHMARK_DIR / "benchmark_summary.csv"
RUNS_PATH = BENCHMARK_DIR / "benchmark_runs.csv"
MODEL_METRICS_PATH = ROOT / "results" / "model_metrics.json"
DATASET_SUMMARY_PATH = ROOT / "data" / "dataset_summary.json"
ALGORITHM_COLOR_SCALE = alt.Scale(
    domain=["Recursive", "Iterative", "ML-guided"],
    range=["#64748B", "#2563EB", "#F97316"],
)

st.set_page_config(
    page_title="N-Queens Research Lab",
    page_icon="♛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
    .hero {padding: 1.2rem 1.4rem; border: 1px solid rgba(128,128,128,.3);
           border-radius: 14px; margin-bottom: 1rem;}
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
    .hero p {margin: 0; opacity: .78;}
    .board {display: grid; gap: 3px; width: fit-content; padding: 5px;
            background: #64748b; border-radius: 8px;}
    .cell {width: 38px; height: 38px; display: flex; align-items: center;
           justify-content: center; font-size: 25px; font-weight: 700;}
    .light {background: #e2e8f0; color: #0f172a;}
    .dark {background: #94a3b8; color: #0f172a;}
    .queen {color: #0f172a; text-shadow: 0 0 1px #fff;}
    </style>
    <div class="hero">
      <h1>♛ N-Queens Research Lab</h1>
      <p>Compare conventional backtracking with Random Forest-guided search.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Research workspace")
page = st.sidebar.radio(
    "Navigate",
    ("Benchmark dashboard", "Solver visualizer", "Model & dataset"),
)
st.sidebar.caption("DAA project · Recursive vs Iterative vs ML-guided")


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_summary() -> pd.DataFrame | None:
    if not SUMMARY_PATH.is_file():
        return None
    try:
        return pd.read_csv(SUMMARY_PATH)
    except (OSError, pd.errors.ParserError):
        return None


def render_board(board: list[int], n: int) -> None:
    cells = []
    for row in range(n):
        for col in range(n):
            occupied = board[row] == col
            shade = "light" if (row + col) % 2 == 0 else "dark"
            content = "♛" if occupied else ""
            queen_class = "queen" if occupied else ""
            cells.append(
                f'<div class="cell {shade} {queen_class}">{content}</div>'
            )
    html = (
        f'<div class="board" style="grid-template-columns: repeat({n}, 38px)">'
        + "".join(cells)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def replay_events(events: list[dict], n: int, step: int) -> list[int]:
    board = [-1] * n
    for event in events[:step]:
        action = event.get("action")
        if action == "place":
            board[event["row"]] = event["col"]
        elif action == "backtrack":
            board[event["row"]] = -1
        elif action == "solution":
            board = list(event["board"])
    return board


if page == "Benchmark dashboard":
    st.header("Benchmark dashboard")
    st.write(
        "Explore repeated-run timing, search effort, and memory measurements "
        "from Phase 6."
    )

    uploaded_summary = st.file_uploader(
        "Use a benchmark_summary.csv file (optional)",
        type=("csv",),
        key="summary_upload",
    )
    if uploaded_summary is not None:
        try:
            summary = pd.read_csv(uploaded_summary)
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            summary = None
    else:
        summary = load_summary()

    if summary is None:
        st.info(
            "No benchmark summary is available yet. Run "
            "'python -m src.benchmark' from the project root, then reopen this page."
        )
    else:
        required_columns = {
            "board_size",
            "algorithm",
            "median_execution_time_ms",
            "mean_execution_time_ms",
            "median_explored_states",
            "median_backtracks",
            "median_candidate_checks",
        }
        missing = required_columns - set(summary.columns)
        if missing:
            st.error("Benchmark CSV is missing columns: " + ", ".join(sorted(missing)))
        else:
            summary["algorithm"] = (
                summary["algorithm"]
                .astype(str)
                .str.lower()
                .map({
                    "recursive": "Recursive",
                    "iterative": "Iterative",
                    "ml": "ML-guided",
                })
                .fillna(summary["algorithm"].astype(str))
            )
            sizes = sorted(summary["board_size"].dropna().astype(int).unique())
            chosen_size = st.selectbox("Board size", sizes, index=0)
            size_frame = summary[summary["board_size"] == chosen_size].copy()

            metric_options = {
                "Median execution time (ms)": "median_execution_time_ms",
                "Mean execution time (ms)": "mean_execution_time_ms",
                "Explored states (median)": "median_explored_states",
                "Backtracks (median)": "median_backtracks",
                "Candidate checks (median)": "median_candidate_checks",
                "Peak traced memory (KiB)": "peak_traced_memory_kb",
                "ML inference time (ms)": "median_inference_time_ms",
            }
            available_metrics = {
                label: column
                for label, column in metric_options.items()
                if column in summary.columns
            }
            selected_label = st.selectbox(
                "Comparison metric", list(available_metrics.keys())
            )
            selected_metric = available_metrics[selected_label]

            metric_values = pd.to_numeric(
                summary[selected_metric], errors="coerce"
            )
            log_scale_allowed = bool(
                metric_values.notna().all() and (metric_values > 0).all()
            )
            scale_options = (
                ["Linear", "Logarithmic"]
                if log_scale_allowed
                else ["Linear"]
            )
            selected_scale = st.radio(
                "Y-axis scale",
                options=scale_options,
                horizontal=True,
                key="benchmark_y_axis_scale",
                help=(
                    "Logarithmic scale makes small and large values easier to compare "
                    "on the same chart. It changes the axis spacing, not the data."
                ),
            )
            if not log_scale_allowed:
                st.caption(
                    "Logarithmic scale is unavailable for this metric because at least "
                    "one algorithm has a zero, negative, or missing value."
                )

            y_axis_scale = (
                alt.Scale(type="log", zero=False)
                if selected_scale == "Logarithmic"
                else alt.Scale(type="linear", zero=True)
            )

            st.subheader(selected_label)
            # One shared set of axes makes every algorithm comparable at the
            # same N and on the same metric scale.
            comparison_chart = (
                alt.Chart(summary)
                .mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=120),
                    strokeWidth=3,
                )
                .encode(
                    x=alt.X(
                        "board_size:Q",
                        title="Board size (N)",
                        axis=alt.Axis(tickMinStep=1),
                    ),
                    y=alt.Y(
                        f"{selected_metric}:Q",
                        title=selected_label,
                        scale=y_axis_scale,
                    ),
                    color=alt.Color(
                        "algorithm:N",
                        title="Algorithm",
                        scale=ALGORITHM_COLOR_SCALE,
                        sort=["Recursive", "Iterative", "ML-guided"],
                    ),
                    strokeDash=alt.StrokeDash(
                        "algorithm:N",
                        title="Line style",
                        scale=alt.Scale(
                            domain=["Recursive", "Iterative", "ML-guided"],
                            range=[[7, 4], [1, 0], [3, 2]],
                        ),
                    ),
                    shape=alt.Shape(
                        "algorithm:N",
                        title="Marker",
                        scale=alt.Scale(
                            domain=["Recursive", "Iterative", "ML-guided"],
                            range=["square", "circle", "triangle"],
                        ),
                    ),
                    detail=alt.Detail("algorithm:N"),
                    order=alt.Order("algorithm:N"),
                    tooltip=[
                        alt.Tooltip("board_size:Q", title="N"),
                        alt.Tooltip("algorithm:N", title="Algorithm"),
                        alt.Tooltip(
                            f"{selected_metric}:Q",
                            title=selected_label,
                            format=".4f",
                        ),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(comparison_chart, width="stretch")

            recursive_values = size_frame.loc[
                size_frame["algorithm"] == "Recursive", selected_metric
            ]
            iterative_values = size_frame.loc[
                size_frame["algorithm"] == "Iterative", selected_metric
            ]
            if (
                not recursive_values.empty
                and not iterative_values.empty
                and float(recursive_values.iloc[0]) == float(iterative_values.iloc[0])
            ):
                st.caption(
                    "Recursive and Iterative overlap where their measured values are "
                    "identical. Both series are included; hover over the shared points "
                    "to inspect the algorithm values."
                )


            st.subheader(f"Results for N={chosen_size}")
            metric_columns = st.columns(3)
            for index, algorithm in enumerate(("Recursive", "Iterative", "ML-guided")):
                match = size_frame[size_frame["algorithm"] == algorithm]
                with metric_columns[index]:
                    st.markdown(f"**{algorithm}**")
                    if match.empty:
                        st.caption("No data")
                    else:
                        record = match.iloc[0]
                        st.metric(
                            "Median runtime",
                            f"{record['median_execution_time_ms']:.4f} ms",
                        )
                        st.metric(
                            "Explored states",
                            f"{record['median_explored_states']:,.0f}",
                        )

            st.subheader("Benchmark summary table")
            st.dataframe(
                summary.sort_values(["board_size", "algorithm"]),
                width="stretch",
                hide_index=True,
            )

            if uploaded_summary is None and RUNS_PATH.is_file():
                with st.expander("Inspect individual timed runs"):
                    try:
                        runs = pd.read_csv(RUNS_PATH)
                        run_algorithms = sorted(runs["algorithm"].unique())
                        selected_algorithms = st.multiselect(
                            "Algorithms",
                            run_algorithms,
                            default=run_algorithms,
                        )
                        st.dataframe(
                            runs[runs["algorithm"].isin(selected_algorithms)],
                            width="stretch",
                            hide_index=True,
                        )
                    except (OSError, pd.errors.ParserError, KeyError) as exc:
                        st.warning(f"Could not read individual runs: {exc}")


elif page == "Solver visualizer":
    st.header("Solver visualizer")
    st.write(
        "Run a solver and move through its recorded events. ML candidate scores "
        "are shown at each state where candidates are ranked."
    )

    controls = st.columns(2)
    with controls[0]:
        n = st.selectbox("Board size", list(range(4, 13)), index=4)
    with controls[1]:
        algorithm = st.selectbox(
            "Algorithm",
            ("Recursive", "Iterative", "ML-guided"),
        )

    run_clicked = st.button("Run solver", type="primary")
    if run_clicked:
        try:
            if algorithm == "Recursive":
                solution, stats, events = solve_n_queens(n)
            elif algorithm == "Iterative":
                solution, stats, events = solve_n_queens_iterative(n)
            else:
                with st.spinner("Loading the trained model and running ML-guided search..."):
                    model = load_trained_model()
                    solution, stats, events = solve_n_queens_ml(n, model=model)

            st.session_state["viz_result"] = {
                "n": n,
                "algorithm": algorithm,
                "solution": solution,
                "stats": stats,
                "events": events,
            }
        except FileNotFoundError as exc:
            st.error(f"{exc} Train the model with 'python -m src.train_model' first.")
        except Exception as exc:
            st.exception(exc)

    result = st.session_state.get("viz_result")
    if result:
        n_result = result["n"]
        events = result["events"]
        stats = result["stats"]
        if result["solution"] is None:
            st.warning(f"No solution exists for N={n_result}.")
        else:
            st.success(
                f"{result['algorithm']} found a valid solution for N={n_result}."
            )

        st.subheader("Search metrics")
        stat_columns = st.columns(4)
        stat_values = (
            ("Explored states", stats.get("explored_states", 0)),
            ("Backtracks", stats.get("backtracks", 0)),
            ("Candidate checks", stats.get("candidate_checks", 0)),
            ("Recorded events", len(events)),
        )
        for column, (label, value) in zip(stat_columns, stat_values):
            column.metric(label, f"{value:,}")

        total_events = len(events)
        step = st.slider(
            "Replay event",
            min_value=0,
            max_value=total_events,
            value=total_events,
            help="0 shows the empty board; the maximum shows the final recorded state.",
        )
        board = replay_events(events, n_result, step)
        current_event = events[step - 1] if step > 0 else None
        board_col, detail_col = st.columns([1, 1.2])
        with board_col:
            st.markdown(f"**Board state · event {step}/{total_events}**")
            render_board(board, n_result)
        with detail_col:
            st.markdown("**Current event**")
            if current_event is None:
                st.caption("Initial empty board")
            else:
                st.json(current_event)

            rankings = [
                event
                for event in events[:step]
                if event.get("action") == "rank_candidates"
            ]
            if rankings:
                latest_ranking = rankings[-1]
                st.markdown(
                    f"**Candidate ranking for row {latest_ranking['row']}**"
                )
                ranking_frame = pd.DataFrame(latest_ranking["candidates"])
                if not ranking_frame.empty:
                    st.dataframe(
                        ranking_frame,
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.caption("No legal candidates at this state.")

        if "inference_seconds" in stats:
            st.caption(
                f"Model inference calls: {stats.get('ml_inference_calls', 0)} · "
                f"Predictions: {stats.get('model_predictions', 0)} · "
                f"predict_proba time: {stats.get('inference_ms', 0):.3f} ms"
            )


else:
    st.header("Model & dataset")
    model_metrics = read_json(MODEL_METRICS_PATH)
    dataset_summary = read_json(DATASET_SUMMARY_PATH)

    if model_metrics:
        st.subheader("Random Forest validation")
        metric_specs = (
            ("Accuracy", "accuracy", True),
            ("Balanced accuracy", "balanced_accuracy", True),
            ("Precision", "precision_positive", False),
            ("Recall", "recall_positive", False),
            ("F1", "f1_positive", False),
            ("ROC-AUC", "roc_auc", False),
            ("PR-AUC", "average_precision_pr_auc", False),
        )
        columns = st.columns(4)
        for index, (label, key, percent) in enumerate(metric_specs):
            value = model_metrics.get(key)
            if value is not None:
                display = f"{value * 100:.2f}%" if percent else f"{value:.4f}"
                columns[index % 4].metric(label, display)
    else:
        st.info("Model metrics are not available in results/model_metrics.json.")

    if dataset_summary:
        st.subheader("Dataset composition")
        dataset_columns = st.columns(4)
        dataset_columns[0].metric(
            "Candidate samples",
            f"{dataset_summary.get('total_candidate_samples', 0):,}",
        )
        dataset_columns[1].metric(
            "Positive samples",
            f"{dataset_summary.get('positive_samples', 0):,}",
        )
        dataset_columns[2].metric(
            "Negative samples",
            f"{dataset_summary.get('negative_samples', 0):,}",
        )
        dataset_columns[3].metric(
            "Feature count",
            len(dataset_summary.get("feature_columns", [])),
        )

        per_size = dataset_summary.get("per_board_size", {})
        if per_size:
            distribution = pd.DataFrame.from_dict(per_size, orient="index")
            distribution.index.name = "board_size"
            distribution.index = distribution.index.astype(int)
            st.bar_chart(
                distribution[["positive_samples", "negative_samples"]].sort_index(),
                width="stretch",
            )
            st.dataframe(
                distribution.sort_index(),
                width="stretch",
            )
    else:
        st.info("Dataset summary is not available in data/dataset_summary.json.")

    st.divider()
    st.subheader("Generate final experimental report")
    st.write(
        "The report generator uses your actual benchmark summary. It does not "
        "invent missing timing results or assume that ML is faster."
    )
    if st.button("Generate Markdown report"):
        try:
            report_path = generate_report()
            report_bytes = report_path.read_bytes()
            st.success("Final report generated from the saved experiment files.")
            st.download_button(
                "Download final_report.md",
                data=report_bytes,
                file_name="final_report.md",
                mime="text/markdown",
            )
            st.markdown(report_bytes.decode("utf-8"))
        except FileNotFoundError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.exception(exc)

st.sidebar.divider()
st.sidebar.caption(
    "Research note: fewer explored states do not guarantee lower runtime. "
    "Compare total elapsed time and search effort separately."
)
