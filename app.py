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
TRIAL_DIR = ROOT / "experiments" / "ml_optimization_trial"
TRIAL_ARCHIVE_PATH = TRIAL_DIR / "source_code.zip"
TRIAL_RUNNER_PATH = TRIAL_DIR / "run_fair_benchmark.py"
TRIAL_TRAINER_PATH = TRIAL_DIR / "train_compact_model.py"
TRIAL_MODEL_PATH = TRIAL_DIR / "results" / "compact_rf.joblib"
TRIAL_OUTPUT_DIR = TRIAL_DIR / "results" / "fair_run"
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
    ("Benchmark dashboard", "Solver visualizer", "All-solutions explorer", "Research extensions", "Model & dataset"),
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
        "Compare the solver approaches using measured benchmark data. "
        "You can use the saved Phase 6 results or run the compiled experiment live."
    )

    with st.container(border=True):
        st.subheader("Run a live three-approach comparison")
        st.caption(
            "The live experiment runs compiled recursive backtracking, compiled "
            "iterative backtracking, and compiled Random Forest-guided backtracking. "
            "Choose any board sizes from 4×4 through 24×24."
        )
        live_size_col, live_repeat_col = st.columns([3, 1])
        with live_size_col:
            live_sizes = st.multiselect(
                "Board sizes to benchmark",
                options=list(range(4, 25)),
                default=list(range(4, 25)),
                help="Select one or more board sizes. Odd sizes are available too.",
                key="live_trial_sizes",
            )
        with live_repeat_col:
            live_repeats = st.number_input(
                "Timed repetitions",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="live_trial_repeats",
            )

        st.caption(
            "The Random Forest is trained on N=4–10. Results for N=11–24 are "
            "extrapolation beyond its training sizes. Compilation and model "
            "training/loading are excluded from the timed solver runs."
        )
        run_live = st.button(
            "Run live benchmark",
            type="primary",
            disabled=not live_sizes,
            key="run_live_compiled_benchmark",
        )

        if run_live:
            try:
                if not TRIAL_RUNNER_PATH.is_file():
                    if not TRIAL_ARCHIVE_PATH.is_file():
                        raise FileNotFoundError(
                            f"Optimisation source archive not found: {TRIAL_ARCHIVE_PATH}"
                        )
                    with zipfile.ZipFile(TRIAL_ARCHIVE_PATH) as archive:
                        bad_member = archive.testzip()
                        if bad_member is not None:
                            raise RuntimeError(
                                f"Source archive CRC check failed for {bad_member}."
                            )
                        root_resolved = TRIAL_DIR.resolve()
                        for member in archive.infolist():
                            destination = (TRIAL_DIR / member.filename).resolve()
                            if (
                                destination != root_resolved
                                and root_resolved not in destination.parents
                            ):
                                raise RuntimeError(
                                    "Unsafe path found in the optimisation source archive."
                                )
                        archive.extractall(TRIAL_DIR)

                if not TRIAL_RUNNER_PATH.is_file():
                    raise FileNotFoundError(
                        "run_fair_benchmark.py was not found inside source_code.zip."
                    )

                if not TRIAL_MODEL_PATH.is_file():
                    with st.spinner(
                        "Training the compact Random Forest for the live benchmark..."
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
                        st.error("Model training failed; the benchmark was not started.")
                        st.code(training.stderr or training.stdout, language="text")
                        st.info(
                            "Install dependencies from the project root with: "
                            "python -m pip install -r "
                            "experiments/ml_optimization_trial/requirements.txt"
                        )
                        raise RuntimeError("Compact model training failed.")

                started_at = time.time()
                measured_at = datetime.now().astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
                command = [
                    sys.executable,
                    str(TRIAL_RUNNER_PATH),
                    "--sizes",
                    *[str(size) for size in sorted(live_sizes)],
                    "--repeats",
                    str(int(live_repeats)),
                ]
                with st.spinner(
                    "Executing and timing the three compiled solvers for each "
                    "selected board size..."
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
                    st.error("The live benchmark failed.")
                    st.code(benchmark.stderr or benchmark.stdout, language="text")
                    st.info(
                        "Check the compiled benchmark dependencies with: "
                        "python -m pip install -r "
                        "experiments/ml_optimization_trial/requirements.txt"
                    )
                else:
                    expected_columns = {
                        "board_size",
                        "recursive_median_ms",
                        "iterative_median_ms",
                        "ml_median_ms",
                    }
                    csv_files = (
                        sorted(
                            TRIAL_OUTPUT_DIR.rglob("*.csv"),
                            key=lambda path: path.stat().st_mtime,
                            reverse=True,
                        )
                        if TRIAL_OUTPUT_DIR.is_dir()
                        else []
                    )
                    fresh_summary = None
                    fresh_path = None
                    for csv_path in csv_files:
                        if csv_path.stat().st_mtime < started_at - 3:
                            continue
                        try:
                            candidate = pd.read_csv(csv_path)
                        except (OSError, pd.errors.ParserError):
                            continue
                        if expected_columns.issubset(candidate.columns):
                            fresh_summary = candidate
                            fresh_path = csv_path
                            break

                    if fresh_summary is None:
                        st.error(
                            "Benchmark completed, but its newly written summary CSV "
                            "could not be located in experiments/ml_optimization_trial/"
                            "results/fair_run."
                        )
                        st.code(benchmark.stdout or "(No benchmark output.)", language="text")
                    else:
                        fresh_summary["board_size"] = pd.to_numeric(
                            fresh_summary["board_size"], errors="coerce"
                        )
                        fresh_summary = fresh_summary[
                            fresh_summary["board_size"].isin(sorted(live_sizes))
                        ].copy()
                        st.session_state["live_trial_wide"] = fresh_summary
                        st.session_state["live_trial_time"] = measured_at
                        st.session_state["live_trial_platform"] = (
                            f"{platform.system()} {platform.release()} · "
                            f"Python {platform.python_version()}"
                        )
                        st.session_state["live_trial_file"] = str(fresh_path)
                        st.session_state["live_trial_stdout"] = benchmark.stdout
                        st.success(
                            "Live benchmark completed. The comparison chart below "
                            "will now use measurements from this run."
                        )
            except subprocess.TimeoutExpired:
                st.error(
                    "The benchmark exceeded the 10-minute limit. Try fewer board "
                    "sizes or fewer repetitions."
                )
            except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as exc:
                st.error(str(exc))

    uploaded_summary = st.file_uploader(
        "Use a benchmark_summary.csv file (optional)",
        type=("csv",),
        key="summary_upload",
    )
    if uploaded_summary is not None:
        try:
            summary = pd.read_csv(uploaded_summary)
            data_source_label = "Uploaded benchmark summary"
        except Exception as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            summary = None
            data_source_label = "Uploaded benchmark summary"
    else:
        live_wide = st.session_state.get("live_trial_wide")
        if isinstance(live_wide, pd.DataFrame) and not live_wide.empty:
            case_values = (
                sorted(live_wide["case"].dropna().astype(str).unique().tolist())
                if "case" in live_wide.columns
                else []
            )
            if case_values:
                preferred_case = (
                    case_values.index("empty_board")
                    if "empty_board" in case_values
                    else 0
                )
                selected_case = st.selectbox(
                    "Live experiment scenario",
                    case_values,
                    index=preferred_case,
                    key="live_trial_case",
                )
                selected_wide = live_wide[
                    live_wide["case"].astype(str) == selected_case
                ].copy()
            else:
                selected_wide = live_wide.copy()

            live_records = []
            algorithm_columns = (
                ("Recursive", "recursive_median_ms"),
                ("Iterative", "iterative_median_ms"),
                ("ML-guided", "ml_median_ms"),
            )
            for algorithm_name, timing_column in algorithm_columns:
                if timing_column not in selected_wide.columns:
                    continue
                for _, measured_row in selected_wide.iterrows():
                    live_records.append(
                        {
                            "board_size": int(measured_row["board_size"]),
                            "algorithm": algorithm_name,
                            "median_execution_time_ms": float(
                                measured_row[timing_column]
                            ),
                        }
                    )
            summary = pd.DataFrame(live_records)
            data_source_label = (
                "Live compiled measurements · "
                f"{st.session_state.get('live_trial_time', 'previous run')}"
            )
            st.caption(
                f"{data_source_label} · "
                f"{st.session_state.get('live_trial_platform', 'local environment')} · "
                f"CSV: {st.session_state.get('live_trial_file', 'results/fair_run')}"
            )
        else:
            summary = load_summary()
            data_source_label = "Saved Phase 6 benchmark summary"

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
            if data_source_label.startswith("Live compiled"):
                st.caption(
                    "This live compiled experiment intentionally compares three "
                    "approaches: recursive, iterative, and ML-guided. MRV remains "
                    "available in the saved original Phase 6 benchmark when present."
                )
            elif "MRV" not in set(summary["algorithm"]):
                st.warning(
                    "This saved benchmark summary predates the MRV baseline. Regenerate it "
                    "with 'python -m src.benchmark' to include all four original algorithms."
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
                if (
                    column in summary.columns
                    and pd.to_numeric(summary[column], errors="coerce").notna().any()
                )
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

            # Explicit legend for the algorithms that are present in this dataset.
            legend_specs = [
                ("Recursive", '<span style="color:#64748B;font-size:1.15rem">■</span> **Recursive** · dashed'),
                ("Iterative", '<span style="color:#2563EB;font-size:1.15rem">●</span> **Iterative** · solid'),
                ("MRV", '<span style="color:#8B5CF6;font-size:1.15rem">◆</span> **MRV** · dashed'),
                ("ML-guided", '<span style="color:#F97316;font-size:1.15rem">▲</span> **ML-guided** · dotted'),
            ]
            present_algorithms = set(summary["algorithm"])
            visible_legends = [
                markup for algorithm, markup in legend_specs
                if algorithm in present_algorithms
            ]
            legend_columns = st.columns(max(1, len(visible_legends)))
            for column, markup in zip(legend_columns, visible_legends):
                column.markdown(markup, unsafe_allow_html=True)

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
            visible_algorithms = [
                algorithm
                for algorithm in ("Recursive", "Iterative", "MRV", "ML-guided")
                if algorithm in set(size_frame["algorithm"])
            ]
            metric_columns = st.columns(max(1, len(visible_algorithms)))
            for index, algorithm in enumerate(visible_algorithms):
                match = size_frame[size_frame["algorithm"] == algorithm]
                with metric_columns[index]:
                    st.markdown(f"**{algorithm}**")
                    if match.empty:
                        st.caption("No data")
                    else:
                        record = match.iloc[0]
                        runtime = record.get("median_execution_time_ms")
                        if pd.notna(runtime):
                            st.metric("Median runtime", f"{runtime:.4f} ms")
                        else:
                            st.metric("Median runtime", "N/A")
                        explored = record.get("median_explored_states")
                        if explored is not None and pd.notna(explored):
                            st.metric("Explored states", f"{explored:,.0f}")
                        elif data_source_label.startswith("Live compiled"):
                            st.caption("Search-state counts are not collected by this timing runner.")

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
