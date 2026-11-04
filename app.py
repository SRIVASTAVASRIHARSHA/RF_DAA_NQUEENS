"""Streamlit dashboard for N-Queens benchmark analysis and solver replay."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from src.iterative_solver import solve_n_queens_iterative
from src.mrv_solver import solve_n_queens_mrv
from src.all_solutions import enumerate_n_queens
from src.ml_guided_solver import load_trained_model, solve_n_queens_ml
from src.recursive_solver import solve_n_queens
from src.reporting import generate_report

ROOT = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "results" / "benchmark"
SUMMARY_PATH = BENCHMARK_DIR / "benchmark_summary.csv"
RUNS_PATH = BENCHMARK_DIR / "benchmark_runs.csv"
MODEL_METRICS_PATH = ROOT / "results" / "model_metrics.json"
DATASET_SUMMARY_PATH = ROOT / "data" / "dataset_summary.json"
RESEARCH_DIR = ROOT / "results" / "research"
OPTIMIZATION_RESULTS_DIR = (
    ROOT / "experiments" / "ml_optimization_trial" / "results"
)
OPTIMIZATION_SUMMARY_PATH = (
    OPTIMIZATION_RESULTS_DIR / "final_fair_benchmark_summary.csv"
)
TRIAL_DIR = ROOT / "experiments" / "ml_optimization_trial"
TRIAL_ARCHIVE_PATH = TRIAL_DIR / "source_code.zip"
TRIAL_RUNNER_PATH = TRIAL_DIR / "run_fair_benchmark.py"
TRIAL_TRAINER_PATH = TRIAL_DIR / "train_compact_model.py"
TRIAL_MODEL_PATH = OPTIMIZATION_RESULTS_DIR / "compact_rf.joblib"
TRIAL_LIVE_OUTPUT_DIR = OPTIMIZATION_RESULTS_DIR / "fair_run"
N24_SWEEP_PATH = (
    OPTIMIZATION_RESULTS_DIR / "n24_all_first_row_columns_summary.csv"
)
ALGORITHM_COLOR_SCALE = alt.Scale(
    domain=["Recursive", "Iterative", "MRV", "ML-guided"],
    range=["#64748B", "#2563EB", "#8B5CF6", "#F97316"],
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
    (
        "Benchmark dashboard",
        "Recorded ML optimisation results (N=8–24)",
        "Solver visualizer",
        "All-solutions explorer",
        "Research extensions",
        "Model & dataset",
    ),
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
        "Run the three comparable compiled approaches live during the presentation. "
        "The graph below is built from measurements produced by the current run."
    )

    st.subheader("Live compiled benchmark · N=4 to N=24")
    st.caption(
        "This experiment runs compiled recursive backtracking, compiled iterative "
        "backtracking, and compiled Random Forest-guided backtracking. It is separate "
        "from the original Python benchmark shown in the collapsed legacy section."
    )

    size_col, repeat_col = st.columns([2, 1])
    with size_col:
        live_sizes = st.multiselect(
            "Board sizes to execute",
            options=list(range(4, 25)),
            default=list(range(4, 25)),
            help=(
                "Choose one board size for a focused live comparison, or several "
                "sizes to build a live performance curve. The maximum is 24×24."
            ),
            key="live_trial_sizes",
        )
    with repeat_col:
        live_repeats = st.number_input(
            "Timed repetitions",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="live_trial_repeats",
        )

    st.caption(
        "The compact Random Forest is trained on N=4–10. Results for larger boards "
        "are an extrapolation of the learned ranking policy. The benchmark warms up "
        "compiled code before timing; JIT compilation and model training/loading are "
        "not included in solver runtime."
    )

    run_live = st.button(
        "Run live benchmark and generate graph",
        type="primary",
        disabled=not live_sizes,
        key="run_live_compiled_benchmark",
    )

    if run_live:
        try:
            if not TRIAL_RUNNER_PATH.is_file():
                if not TRIAL_ARCHIVE_PATH.is_file():
                    raise FileNotFoundError(
                        "The optimisation source archive is missing: "
                        f"{TRIAL_ARCHIVE_PATH}"
                    )
                with zipfile.ZipFile(TRIAL_ARCHIVE_PATH) as archive:
                    bad_member = archive.testzip()
                    if bad_member is not None:
                        raise RuntimeError(
                            "The optimisation source archive failed its CRC check "
                            f"at {bad_member}."
                        )
                    root_resolved = TRIAL_DIR.resolve()
                    for member in archive.infolist():
                        destination = (TRIAL_DIR / member.filename).resolve()
                        if destination != root_resolved and root_resolved not in destination.parents:
                            raise RuntimeError(
                                "Unsafe path found in the optimisation source archive."
                            )
                    archive.extractall(TRIAL_DIR)

            if not TRIAL_RUNNER_PATH.is_file():
                raise FileNotFoundError(
                    "run_fair_benchmark.py was not present after extracting "
                    "source_code.zip."
                )

            if not TRIAL_MODEL_PATH.is_file():
                with st.spinner(
                    "Training the compact Random Forest because no trained trial "
                    "model was found..."
                ):
                    training = subprocess.run(
                        [sys.executable, str(TRIAL_TRAINER_PATH)],
                        cwd=str(TRIAL_DIR),
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=300,
                    )
                if training.returncode != 0:
                    st.error("The compact model could not be trained.")
                    st.code(training.stderr or training.stdout, language="text")
                    st.info(
                        "Install the experiment dependencies from the project root "
                        "with: python -m pip install -r "
                        "experiments/ml_optimization_trial/requirements.txt"
                    )
                    raise RuntimeError("Model training failed; benchmark was not run.")

            command = [
                sys.executable,
                str(TRIAL_RUNNER_PATH),
                "--sizes",
                *[str(size) for size in sorted(live_sizes)],
                "--repeats",
                str(int(live_repeats)),
            ]
            started_at = time.time()
            started_label = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            with st.spinner(
                "Running all three compiled solvers, validating solutions, and "
                "measuring each selected board size..."
            ):
                benchmark = subprocess.run(
                    command,
                    cwd=str(TRIAL_DIR),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )

            if benchmark.returncode != 0:
                st.error("The live benchmark process failed.")
                st.code(benchmark.stderr or benchmark.stdout, language="text")
                st.info(
                    "Check that Numba and the experiment packages are installed: "
                    "python -m pip install -r "
                    "experiments/ml_optimization_trial/requirements.txt"
                )
            else:
                required_live_columns = {
                    "board_size",
                    "recursive_median_ms",
                    "iterative_median_ms",
                    "ml_median_ms",
                }
                candidates = (
                    sorted(
                        TRIAL_LIVE_OUTPUT_DIR.rglob("*.csv"),
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
                    if TRIAL_LIVE_OUTPUT_DIR.is_dir()
                    else []
                )
                live_summary = None
                live_summary_path = None
                for candidate in candidates:
                    try:
                        candidate_frame = pd.read_csv(candidate)
                    except (OSError, pd.errors.ParserError):
                        continue
                    if required_live_columns.issubset(candidate_frame.columns):
                        if candidate.stat().st_mtime >= started_at - 3:
                            live_summary = candidate_frame
                            live_summary_path = candidate
                            break

                if live_summary is None:
                    st.error(
                        "The benchmark finished, but no fresh summary CSV with the "
                        "expected timing columns was found in results/fair_run. "
                        "Review the command output below."
                    )
                    st.code(benchmark.stdout or "(No standard output.)", language="text")
                else:
                    live_summary["board_size"] = pd.to_numeric(
                        live_summary["board_size"], errors="coerce"
                    )
                    live_summary = live_summary[
                        live_summary["board_size"].isin(sorted(live_sizes))
                    ].copy()
                    st.session_state["live_trial_summary"] = live_summary
                    st.session_state["live_trial_timestamp"] = started_label
                    st.session_state["live_trial_output"] = str(live_summary_path)
                    st.session_state["live_trial_stdout"] = benchmark.stdout
                    st.session_state["live_trial_platform"] = (
                        f"{platform.system()} {platform.release()} · "
                        f"Python {platform.python_version()}"
                    )
                    st.success(
                        "Live benchmark completed. The chart below uses the newly "
                        "measured results from this run."
                    )
        except subprocess.TimeoutExpired:
            st.error(
                "The benchmark exceeded the 10-minute safety limit. Try fewer board "
                "sizes or fewer repetitions."
            )
        except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
            st.error(str(exc))

    live_result = st.session_state.get("live_trial_summary")
    if isinstance(live_result, pd.DataFrame) and not live_result.empty:
        live_result = live_result.sort_values("board_size").copy()
        timestamp = st.session_state.get("live_trial_timestamp", "Previous run")
        st.caption(
            f"Measured live at {timestamp} · "
            f"{st.session_state.get('live_trial_platform', 'Local environment')} · "
            f"Source: {st.session_state.get('live_trial_output', 'fair_run summary')}"
        )

        case_values = (
            sorted(live_result["case"].dropna().astype(str).unique().tolist())
            if "case" in live_result.columns
            else []
        )
        if len(case_values) > 1:
            selected_case = st.selectbox(
                "Benchmark case",
                case_values,
                key="live_trial_case",
            )
            plotted_result = live_result[
                live_result["case"].astype(str) == selected_case
            ].copy()
        else:
            plotted_result = live_result.copy()
            if case_values:
                st.caption(f"Benchmark case: {case_values[0]}")

        chart_columns = [
            "recursive_median_ms",
            "iterative_median_ms",
            "ml_median_ms",
        ]
        chart_frame = plotted_result.melt(
            id_vars=["board_size"],
            value_vars=chart_columns,
            var_name="approach",
            value_name="median_runtime_ms",
        )
        chart_frame["approach"] = chart_frame["approach"].map(
            {
                "recursive_median_ms": "Compiled recursive",
                "iterative_median_ms": "Compiled iterative",
                "ml_median_ms": "Compiled ML-guided",
            }
        )
        live_chart = (
            alt.Chart(chart_frame)
            .mark_line(point=alt.OverlayMarkDef(filled=True, size=100))
            .encode(
                x=alt.X(
                    "board_size:Q",
                    title="Board size (N)",
                    axis=alt.Axis(tickMinStep=1),
                ),
                y=alt.Y(
                    "median_runtime_ms:Q",
                    title="Median runtime (ms)",
                    scale=alt.Scale(type="log", zero=False),
                ),
                color=alt.Color("approach:N", title="Approach"),
                tooltip=[
                    alt.Tooltip("board_size:Q", title="Board size"),
                    alt.Tooltip("approach:N", title="Approach"),
                    alt.Tooltip(
                        "median_runtime_ms:Q",
                        title="Median runtime (ms)",
                        format=".6f",
                    ),
                ],
            )
            .properties(height=410)
        )
        st.subheader("Live runtime comparison")
        st.altair_chart(live_chart, width="stretch")
        st.caption(
            "The plotted points come from the benchmark run just completed above. "
            "The logarithmic Y-axis helps compare runtimes across different scales."
        )

        available_live_sizes = sorted(
            plotted_result["board_size"].dropna().astype(int).unique().tolist()
        )
        if available_live_sizes:
            inspect_n = st.selectbox(
                "Inspect measured board size",
                available_live_sizes,
                index=len(available_live_sizes) - 1,
                key="live_trial_inspect_n",
            )
            selected_row = plotted_result[
                plotted_result["board_size"].astype(int) == inspect_n
            ].iloc[-1]
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                "Compiled recursive",
                f"{selected_row['recursive_median_ms']:.6f} ms",
            )
            metric_columns[1].metric(
                "Compiled iterative",
                f"{selected_row['iterative_median_ms']:.6f} ms",
            )
            metric_columns[2].metric(
                "Compiled ML-guided",
                f"{selected_row['ml_median_ms']:.6f} ms",
            )

            recursive_speedup = (
                float(selected_row["recursive_median_ms"])
                / float(selected_row["ml_median_ms"])
            )
            iterative_speedup = (
                float(selected_row["iterative_median_ms"])
                / float(selected_row["ml_median_ms"])
            )
            speedup_columns = st.columns(2)
            speedup_columns[0].metric(
                "ML speedup vs recursive",
                f"{recursive_speedup:.2f}×",
                help="Above 1× means ML-guided runtime was lower in this run.",
            )
            speedup_columns[1].metric(
                "ML speedup vs iterative",
                f"{iterative_speedup:.2f}×",
                help="Above 1× means ML-guided runtime was lower in this run.",
            )

        st.subheader("Measured data from this run")
        st.dataframe(plotted_result, width="stretch", hide_index=True)
        with st.expander("Show benchmark command output"):
            st.code(
                st.session_state.get("live_trial_stdout", "(No output captured.)"),
                language="text",
            )

    st.divider()
    st.subheader("Legacy Python benchmark · original implementation (N≤12)")
    st.caption(
        "This section reads the original project's benchmark_summary.csv. "
        "It is retained for the earlier Python implementation and is not the "
        "live compiled three-approach experiment above."
    )
    with st.expander("Open original benchmark dashboard", expanded=False):
        # Legacy benchmark content remains available below.

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
                        "mrv": "MRV",
                        "ml": "ML-guided",
                    })
                    .fillna(summary["algorithm"].astype(str))
                )
                if "MRV" not in set(summary["algorithm"]):
                    st.warning(
                        "This benchmark summary predates the MRV baseline. Regenerate it "
                        "with 'python -m src.benchmark' to include all four algorithms."
                    )
                else:
                    st.caption(
                        "MRV assigns rows out of order and recomputes legal domains across "
                        "remaining rows. Its candidate-check count is therefore not directly "
                        "equivalent to fixed-row candidate checks; use total runtime as the "
                        "direct cost comparison."
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
                default_log_metrics = {
                    "median_execution_time_ms",
                    "mean_execution_time_ms",
                    "peak_traced_memory_kb",
                    "median_inference_time_ms",
                }
                default_scale_index = (
                    1
                    if log_scale_allowed and selected_metric in default_log_metrics
                    else 0
                )
                selected_scale = st.radio(
                    "Y-axis scale",
                    options=scale_options,
                    index=default_scale_index,
                    horizontal=True,
                    key="benchmark_y_axis_scale_v2",
                    help=(
                        "Logarithmic scale makes small and large positive values easier "
                        "to compare on one chart. It changes axis spacing, not the data."
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
                # Draw Iterative first, then ML, then Recursive. The dashed Recursive
                # stroke sits above the Iterative stroke when their values coincide,
                # while the hollow circle/square markers let both remain identifiable.
                base_chart = alt.Chart(summary).encode(
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
                    tooltip=[
                        alt.Tooltip("board_size:Q", title="N"),
                        alt.Tooltip("algorithm:N", title="Algorithm"),
                        alt.Tooltip(
                            f"{selected_metric}:Q",
                            title=selected_label,
                            format=".6f",
                        ),
                    ],
                )
    
                iterative_data = base_chart.transform_filter(
                    alt.datum.algorithm == "Iterative"
                )
                recursive_data = base_chart.transform_filter(
                    alt.datum.algorithm == "Recursive"
                )
                mrv_data = base_chart.transform_filter(
                    alt.datum.algorithm == "MRV"
                )
                ml_data = base_chart.transform_filter(
                    alt.datum.algorithm == "ML-guided"
                )
    
                iterative_line = iterative_data.mark_line(
                    color="#2563EB",
                    strokeWidth=3,
                )
                mrv_line = mrv_data.mark_line(
                    color="#8B5CF6",
                    strokeWidth=3,
                    strokeDash=[5, 3],
                )
                ml_line = ml_data.mark_line(
                    color="#F97316",
                    strokeWidth=3,
                    strokeDash=[3, 2],
                )
                recursive_line = recursive_data.mark_line(
                    color="#64748B",
                    strokeWidth=3.5,
                    strokeDash=[7, 4],
                )
    
                iterative_points = iterative_data.mark_point(
                    shape="circle",
                    filled=False,
                    size=210,
                    strokeWidth=3,
                    color="#2563EB",
                )
                mrv_points = mrv_data.mark_point(
                    shape="diamond",
                    filled=False,
                    size=190,
                    strokeWidth=3,
                    color="#8B5CF6",
                )
                ml_points = ml_data.mark_point(
                    shape="triangle",
                    filled=False,
                    size=190,
                    strokeWidth=3,
                    color="#F97316",
                )
                recursive_points = recursive_data.mark_point(
                    shape="square",
                    filled=False,
                    size=125,
                    strokeWidth=3,
                    color="#64748B",
                )
    
                comparison_chart = alt.layer(
                    iterative_line,
                    mrv_line,
                    ml_line,
                    recursive_line,
                    iterative_points,
                    mrv_points,
                    ml_points,
                    recursive_points,
                ).properties(height=420)
                st.altair_chart(comparison_chart, width="stretch")
    
                # Explicit legend for the separately layered algorithm series.
                legend_columns = st.columns(4)
                legend_columns[0].markdown(
                    '<span style="color:#64748B;font-size:1.15rem">■</span> **Recursive** · dashed',
                    unsafe_allow_html=True,
                )
                legend_columns[1].markdown(
                    '<span style="color:#2563EB;font-size:1.15rem">●</span> **Iterative** · solid',
                    unsafe_allow_html=True,
                )
                legend_columns[2].markdown(
                    '<span style="color:#8B5CF6;font-size:1.15rem">◆</span> **MRV** · dashed',
                    unsafe_allow_html=True,
                )
                legend_columns[3].markdown(
                    '<span style="color:#F97316;font-size:1.15rem">▲</span> **ML-guided** · dotted',
                    unsafe_allow_html=True,
                )
    
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
                        "Recursive and Iterative have identical values for this board size. "
                        "Their strokes overlap by definition; the dashed line and hollow "
                        "square/circle markers distinguish the two without changing data."
                    )
    
                st.subheader(f"Results for N={chosen_size}")
                metric_columns = st.columns(4)
                for index, algorithm in enumerate(("Recursive", "Iterative", "MRV", "ML-guided")):
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
        st.caption(
            "The original interactive solvers support N≤12. Larger-board timing "
            "results are available in the separate ML optimisation trial page; "
            "that compiled experiment is not wired into this event-by-event visualizer."
        )
    
        controls = st.columns(2)
        with controls[0]:
            n = st.selectbox("Board size", list(range(4, 13)), index=4)
        with controls[1]:
            algorithm = st.selectbox(
                "Algorithm",
                ("Recursive", "Iterative", "MRV", "ML-guided"),
            )
    
        run_clicked = st.button("Run solver", type="primary")
        if run_clicked:
            try:
                if algorithm == "Recursive":
                    solution, stats, events = solve_n_queens(n)
                elif algorithm == "Iterative":
                    solution, stats, events = solve_n_queens_iterative(n)
                elif algorithm == "MRV":
                    solution, stats, events = solve_n_queens_mrv(n)
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
    

elif page == "Recorded ML optimisation results (N=8–24)":
    st.header("ML optimisation trial · Larger board sizes")
    st.write(
        "This view presents the separate compiled optimisation experiment. "
        "It does not merge its timings with the original Python benchmark, "
        "because the implementations and execution environment differ."
    )

    if not OPTIMIZATION_SUMMARY_PATH.is_file():
        st.error(
            "The committed optimisation summary CSV is missing. Pull the latest "
            "main branch and check experiments/ml_optimization_trial/results/."
        )
    else:
        try:
            trial = pd.read_csv(OPTIMIZATION_SUMMARY_PATH)
            required = {
                "board_size",
                "case",
                "runs",
                "recursive_median_ms",
                "iterative_median_ms",
                "ml_median_ms",
            }
            missing = required - set(trial.columns)
            if missing:
                st.error(
                    "Optimisation summary is missing columns: "
                    + ", ".join(sorted(missing))
                )
            else:
                trial = trial.sort_values("board_size").copy()
                trial["board_size"] = trial["board_size"].astype(int)
                trial_sizes = sorted(trial["board_size"].unique().tolist())

                st.success(
                    f"Loaded compiled benchmark results for N={trial_sizes[0]} "
                    f"through N={trial_sizes[-1]}."
                )
                st.caption(
                    "These are the recorded experiment results, not fresh timings "
                    "from the computer currently running this dashboard."
                )

                chart_frame = trial.melt(
                    id_vars=["board_size", "runs"],
                    value_vars=[
                        "recursive_median_ms",
                        "iterative_median_ms",
                        "ml_median_ms",
                    ],
                    var_name="solver",
                    value_name="median_ms",
                )
                chart_frame["solver"] = chart_frame["solver"].map(
                    {
                        "recursive_median_ms": "Compiled recursive",
                        "iterative_median_ms": "Compiled iterative",
                        "ml_median_ms": "Compiled ML-guided",
                    }
                )
                timing_chart = (
                    alt.Chart(chart_frame)
                    .mark_line(point=alt.OverlayMarkDef(filled=True, size=85))
                    .encode(
                        x=alt.X(
                            "board_size:Q",
                            title="Board size (N)",
                            axis=alt.Axis(tickMinStep=1),
                        ),
                        y=alt.Y(
                            "median_ms:Q",
                            title="Median runtime (ms)",
                            scale=alt.Scale(type="log", zero=False),
                        ),
                        color=alt.Color("solver:N", title="Implementation"),
                        tooltip=[
                            alt.Tooltip("board_size:Q", title="N"),
                            alt.Tooltip("solver:N", title="Implementation"),
                            alt.Tooltip("median_ms:Q", title="Median runtime (ms)", format=".6f"),
                            alt.Tooltip("runs:Q", title="Timed runs"),
                        ],
                    )
                    .properties(height=400)
                )
                st.subheader("Empty-board runtime across board sizes")
                st.altair_chart(timing_chart, width="stretch")
                st.caption(
                    "Logarithmic Y-axis is used because runtime spans several "
                    "orders of magnitude. Each point is the median of the run "
                    "count shown in the tooltip."
                )

                selected_n = st.selectbox(
                    "Inspect board size",
                    trial_sizes,
                    index=len(trial_sizes) - 1,
                    key="optimization_trial_board_size",
                )
                selected = trial[trial["board_size"] == selected_n].iloc[0]
                metric_cols = st.columns(4)
                metric_cols[0].metric("Board size", f"N={selected_n}")
                metric_cols[1].metric(
                    "Compiled recursive",
                    f"{selected['recursive_median_ms']:.4f} ms",
                )
                metric_cols[2].metric(
                    "Compiled iterative",
                    f"{selected['iterative_median_ms']:.4f} ms",
                )
                metric_cols[3].metric(
                    "Compiled ML-guided",
                    f"{selected['ml_median_ms']:.4f} ms",
                )

                recursive_speedup = (
                    selected["recursive_median_ms"] / selected["ml_median_ms"]
                )
                iterative_speedup = (
                    selected["iterative_median_ms"] / selected["ml_median_ms"]
                )
                speedup_cols = st.columns(2)
                speedup_cols[0].metric(
                    "ML speedup vs recursive",
                    f"{recursive_speedup:.2f}×",
                    help="Above 1× means ML-guided runtime was lower in this recorded case.",
                )
                speedup_cols[1].metric(
                    "ML speedup vs iterative",
                    f"{iterative_speedup:.2f}×",
                    help="Above 1× means ML-guided runtime was lower in this recorded case.",
                )

                st.subheader("Recorded benchmark data")
                st.dataframe(trial, width="stretch", hide_index=True)

        except (OSError, pd.errors.ParserError, KeyError, ValueError) as exc:
            st.error(f"Could not load the optimisation summary: {exc}")

    st.divider()
    st.subheader("N=24 · First-row queen placement sweep")
    st.write(
        "This experiment fixes the queen in row 0, tests each of the 24 possible "
        "columns, and compares the compiled implementations. It is separate from "
        "the empty-board benchmark above."
    )

    if not N24_SWEEP_PATH.is_file():
        st.info("The N=24 first-row sweep CSV is not present in the repository.")
    else:
        try:
            sweep = pd.read_csv(N24_SWEEP_PATH)
            sweep_required = {
                "fixed_first_row_col",
                "speedup_vs_recursive",
                "speedup_vs_iterative",
            }
            sweep_missing = sweep_required - set(sweep.columns)
            if sweep_missing:
                st.error(
                    "N=24 sweep CSV is missing columns: "
                    + ", ".join(sorted(sweep_missing))
                )
            else:
                faster_recursive = int((sweep["speedup_vs_recursive"] > 1).sum())
                faster_iterative = int((sweep["speedup_vs_iterative"] > 1).sum())
                median_recursive = float(sweep["speedup_vs_recursive"].median())
                median_iterative = float(sweep["speedup_vs_iterative"].median())

                sweep_cols = st.columns(4)
                sweep_cols[0].metric(
                    "Columns faster than recursive",
                    f"{faster_recursive}/24",
                )
                sweep_cols[1].metric(
                    "Columns faster than iterative",
                    f"{faster_iterative}/24",
                )
                sweep_cols[2].metric(
                    "Median speedup vs recursive",
                    f"{median_recursive:.2f}×",
                )
                sweep_cols[3].metric(
                    "Median speedup vs iterative",
                    f"{median_iterative:.2f}×",
                )

                sweep_long = sweep.melt(
                    id_vars=["fixed_first_row_col"],
                    value_vars=[
                        "speedup_vs_recursive",
                        "speedup_vs_iterative",
                    ],
                    var_name="comparison",
                    value_name="speedup",
                )
                sweep_long["comparison"] = sweep_long["comparison"].map(
                    {
                        "speedup_vs_recursive": "vs compiled recursive",
                        "speedup_vs_iterative": "vs compiled iterative",
                    }
                )
                sweep_chart = (
                    alt.Chart(sweep_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X(
                            "fixed_first_row_col:Q",
                            title="Fixed queen column in row 0",
                            axis=alt.Axis(tickMinStep=1),
                        ),
                        y=alt.Y(
                            "speedup:Q",
                            title="ML speedup ratio (above 1× = ML faster)",
                            scale=alt.Scale(type="log", zero=False),
                        ),
                        color=alt.Color("comparison:N", title="Comparison"),
                        tooltip=[
                            alt.Tooltip("fixed_first_row_col:Q", title="Fixed column"),
                            alt.Tooltip("comparison:N", title="Comparison"),
                            alt.Tooltip("speedup:Q", title="Speedup", format=".3f"),
                        ],
                    )
                    .properties(height=340)
                )
                baseline = (
                    alt.Chart(pd.DataFrame({"baseline": [1.0]}))
                    .mark_rule(strokeDash=[5, 4], color="#64748B")
                    .encode(y="baseline:Q")
                )
                st.altair_chart(sweep_chart + baseline, width="stretch")
                st.dataframe(sweep, width="stretch", hide_index=True)
                st.caption(
                    "The result is not universal: ML was faster in 18/24 positions "
                    "versus compiled recursive and 19/24 versus compiled iterative. "
                    "A fixed first-row queen does not represent arbitrary pre-placed "
                    "queens or every possible search state."
                )
        except (OSError, pd.errors.ParserError, KeyError, ValueError) as exc:
            st.error(f"Could not load the N=24 sweep: {exc}")

    with st.expander("How to interpret these results"):
        st.markdown(
            """
            - The optimisation trial uses a compact Random Forest and compiled
              feature extraction/tree traversal. Its recursive and iterative
              baselines are also compiled.
            - The model was trained and validated on N=4–10. N=14–24 results are
              extrapolation beyond the training board sizes.
            - The recorded measurements were collected on Linux x86_64 with
              Python 3.13.5. They are not MacBook Air M5 timings.
            - This benchmark page reads saved experiment CSVs. It does not run
              the experimental solver interactively.
            """
        )


elif page == "All-solutions explorer":
    st.header("All-solutions explorer")
    st.write(
        "Enumerate multiple valid N-Queens boards instead of stopping at the first. "
        "Enumeration can grow rapidly, so use a solution cap for larger N."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        enum_n = st.selectbox("Board size for enumeration", list(range(4, 13)), index=4)
    with col_b:
        solution_cap = st.number_input(
            "Maximum solutions to collect",
            min_value=1,
            max_value=10000,
            value=100,
            step=10,
        )
    if st.button("Enumerate solutions", type="primary"):
        with st.spinner("Enumerating valid queen placements..."):
            solutions, enum_stats, _ = enumerate_n_queens(
                enum_n,
                max_solutions=int(solution_cap),
                collect_events=False,
            )
        st.session_state["all_solutions_result"] = {
            "n": enum_n,
            "solutions": solutions,
            "stats": enum_stats,
        }

    enum_result = st.session_state.get("all_solutions_result")
    if enum_result:
        enum_stats = enum_result["stats"]
        summary_cols = st.columns(4)
        summary_cols[0].metric("Solutions found", f"{enum_stats['solutions_found']:,}")
        summary_cols[1].metric("Explored states", f"{enum_stats['explored_states']:,}")
        summary_cols[2].metric("Candidate checks", f"{enum_stats['candidate_checks']:,}")
        summary_cols[3].metric("Backtracks", f"{enum_stats['backtracks']:,}")
        if enum_stats["truncated"]:
            st.warning(
                "The solution cap was reached. The search stopped at the cap; "
                "additional solutions may remain unexplored."
            )
        else:
            st.success("The search exhausted the solution tree.")
        st.write(
            f"Showing up to the first 12 boards from {len(enum_result['solutions']):,} collected solutions."
        )
        for index, solution in enumerate(enum_result["solutions"][:12], start=1):
            with st.expander(f"Solution {index}"):
                render_board(solution, enum_result["n"])


elif page == "Research extensions":
    st.header("Research extensions")
    st.write(
        "Additional research experiments: classifier comparison, feature ablation, "
        "and generalization to board sizes excluded from training."
    )
    st.code(
        "python -m src.research_experiments",
        language="bash",
    )
    st.caption(
        "Default experiment: train on N=4–8 and evaluate on grouped validation "
        "states plus unseen N=9 and N=10. Outputs are saved under results/research/."
    )

    comparison_path = RESEARCH_DIR / "model_comparison.csv"
    ablation_path = RESEARCH_DIR / "feature_ablation.csv"
    generalization_path = RESEARCH_DIR / "unseen_size_generalization.csv"

    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        st.subheader("Classifier comparison")
        st.dataframe(comparison, width="stretch", hide_index=True)
        if {"model", "balanced_accuracy"}.issubset(comparison.columns):
            st.bar_chart(
                comparison.set_index("model")[["balanced_accuracy"]],
                width="stretch",
            )
    else:
        st.info("Classifier comparison is not available yet. Run the command above.")

    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path)
        st.subheader("Feature ablation")
        st.dataframe(ablation, width="stretch", hide_index=True)
        if {"model", "balanced_accuracy"}.issubset(ablation.columns):
            st.bar_chart(
                ablation.set_index("model")[["balanced_accuracy"]],
                width="stretch",
            )
    else:
        st.info("Feature ablation results are not available yet.")

    if generalization_path.is_file():
        generalization = pd.read_csv(generalization_path)
        st.subheader("Unseen-size generalization")
        st.dataframe(generalization, width="stretch", hide_index=True)
        if {"board_size", "balanced_accuracy"}.issubset(generalization.columns):
            plotted = generalization[
                generalization.get("status", "evaluated") == "evaluated"
            ]
            if not plotted.empty:
                st.line_chart(
                    plotted.set_index("board_size")[["balanced_accuracy"]],
                    width="stretch",
                )
    else:
        st.info("Unseen board-size evaluation has not been generated yet.")


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

    feature_importance_path = ROOT / "results" / "feature_importance.csv"
    confusion_matrix_path = ROOT / "results" / "confusion_matrix.png"

    if feature_importance_path.is_file():
        st.subheader("Random Forest feature importance")
        try:
            feature_importance = pd.read_csv(feature_importance_path)
            if {"feature", "importance"}.issubset(feature_importance.columns):
                top_features = feature_importance.sort_values(
                    "importance", ascending=False
                ).head(15)
                st.bar_chart(
                    top_features.set_index("feature")[["importance"]],
                    width="stretch",
                )
                st.dataframe(top_features, width="stretch", hide_index=True)
        except (OSError, pd.errors.ParserError) as exc:
            st.warning(f"Could not load feature importance: {exc}")

    if confusion_matrix_path.is_file():
        st.subheader("Validation confusion matrix")
        st.image(str(confusion_matrix_path), width="stretch")

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
