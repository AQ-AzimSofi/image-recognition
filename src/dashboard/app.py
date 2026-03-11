"""
POC Comparison Dashboard: Object Recognition Pipeline Comparison

Run with: streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")

import streamlit as st
from PIL import Image
from pillow_heif import register_heif_opener
import concurrent.futures

register_heif_opener()

from src.pipelines.registry import PIPELINE_REGISTRY, get_pipeline, list_pipelines
from src.pipelines.base import PipelineResult
from src.providers.base import DetectionBox
from src.dashboard.drawing import draw_detections, CATEGORY_COLORS
from src.dashboard.credentials import check_all_credentials, get_available_pipelines

RESULTS_DIR = _project_root / "results"

st.set_page_config(
    page_title="Vision Pipeline POC - Object Recognition",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = ["upload", "configure", "results"]
STEP_LABELS = {"upload": "1. Upload Images", "configure": "2. Configure Pipelines", "results": "3. Results"}
TMP_DIR = Path("/tmp/vision_poc")


def _save_results(label: str, results: dict[str, list[PipelineResult]], image_files: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.json"
    filepath = RESULTS_DIR / filename

    data = {
        "label": label,
        "timestamp": timestamp,
        "images": image_files,
        "results": {
            image_name: [asdict(r) for r in result_list]
            for image_name, result_list in results.items()
        },
    }
    filepath.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return filepath


def _load_saved_results(filepath: Path) -> dict:
    data = json.loads(filepath.read_text(encoding="utf-8"))
    restored: dict[str, list[PipelineResult]] = {}
    for image_name, result_dicts in data["results"].items():
        result_list = []
        for rd in result_dicts:
            boxes = [DetectionBox(**b) for b in rd.get("boxes", [])]
            result_list.append(PipelineResult(
                pipeline_name=rd["pipeline_name"],
                category=rd["category"],
                boxes=boxes,
                latency_ms=rd.get("latency_ms", 0),
                cost_estimate=rd.get("cost_estimate", 0),
                raw_responses=rd.get("raw_responses", []),
                error=rd.get("error"),
            ))
        restored[image_name] = result_list
    data["results"] = restored
    return data


def _list_saved_results() -> list[dict]:
    if not RESULTS_DIR.exists():
        return []
    entries = []
    for f in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            entries.append({
                "path": f,
                "label": raw.get("label", ""),
                "timestamp": raw.get("timestamp", f.stem),
                "image_count": len(raw.get("images", [])),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return entries


def _delete_saved_result(filepath: Path):
    if filepath.exists():
        filepath.unlink()


def _init_state():
    defaults = {
        "step": "upload",
        "image_files": [],
        "selected_keys": [],
        "results": {},
        "run_parallel": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go_to(step: str):
    st.session_state.step = step


def main():
    _init_state()

    st.title("Object Recognition POC")

    _render_sidebar()

    current = st.session_state.step

    _render_step_indicator(current)

    if current == "upload":
        _page_upload()
    elif current == "configure":
        _page_configure()
    elif current == "results":
        _page_results()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.header("Navigation")
        for step_key, step_label in STEP_LABELS.items():
            is_current = st.session_state.step == step_key
            if is_current:
                st.markdown(f"**>> {step_label}**")
            else:
                if st.button(step_label, key=f"nav_{step_key}", use_container_width=True):
                    _go_to(step_key)

        st.divider()

        if st.button("Check API Credentials", use_container_width=True):
            st.session_state.show_creds = True

        if st.session_state.get("show_creds"):
            _render_credential_check()

        st.divider()
        _render_saved_results_sidebar()

        st.divider()
        n_images = len(st.session_state.image_files)
        n_pipelines = len(st.session_state.selected_keys)
        st.caption(f"Images: {n_images} | Pipelines: {n_pipelines}")
        if n_images > 0 and n_pipelines > 0:
            st.caption(f"Total runs: {n_images * n_pipelines}")


def _render_saved_results_sidebar():
    st.header("Saved Results")
    saved = _list_saved_results()
    if not saved:
        st.caption("No saved results yet.")
        return

    for i, entry in enumerate(saved):
        display = entry["label"] or entry["timestamp"]
        col_load, col_del = st.columns([3, 1])
        with col_load:
            if st.button(
                f"{display} ({entry['image_count']} img)",
                key=f"load_{i}",
                use_container_width=True,
            ):
                data = _load_saved_results(entry["path"])
                st.session_state.results = data["results"]
                st.session_state.image_files = data.get("images", [])
                st.session_state.step = "results"
                st.rerun()
        with col_del:
            if st.button("x", key=f"del_{i}"):
                _delete_saved_result(entry["path"])
                st.rerun()

    st.caption(f"{len(saved)} saved comparison(s)")


def _render_credential_check():
    results = check_all_credentials()
    for cred in results:
        if cred["ok"]:
            st.success(f"{cred['display']}")
            for d in cred["details"]:
                st.caption(f"  {d['label']}: {d['message']}")
        else:
            st.error(f"{cred['display']}")
            for d in cred["details"]:
                if d["ok"]:
                    st.caption(f"  {d['label']}: {d['message']}")
                else:
                    st.caption(f"  {d['label']}: NOT SET")

    available = get_available_pipelines()
    all_keys = set(PIPELINE_REGISTRY.keys())
    unavailable = all_keys - available

    if unavailable:
        st.warning(f"Unavailable pipelines: {', '.join(sorted(unavailable))}")
    else:
        st.success("All pipelines are available")

    if st.button("Close", key="close_creds"):
        st.session_state.show_creds = False
        st.rerun()


def _render_step_indicator(current: str):
    cols = st.columns(3)
    for i, (step_key, step_label) in enumerate(STEP_LABELS.items()):
        with cols[i]:
            if step_key == current:
                st.markdown(
                    f'<div style="text-align:center;padding:8px;background:#0d6efd;'
                    f'color:white;border-radius:8px;font-weight:bold">{step_label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="text-align:center;padding:8px;background:#e9ecef;'
                    f'color:#6c757d;border-radius:8px">{step_label}</div>',
                    unsafe_allow_html=True,
                )
    st.markdown("")


# ---------------------------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------------------------

def _page_upload():
    st.subheader("Upload Test Images")
    st.caption("Upload one or more images to compare across pipelines.")

    uploaded = st.file_uploader(
        "Select images",
        type=["jpg", "jpeg", "png", "webp", "heic"],
        accept_multiple_files=True,
        key="uploader",
        help="Select multiple files at once or upload them one by one",
    )

    if uploaded:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in uploaded:
            dest = TMP_DIR / f.name
            dest.write_bytes(f.getvalue())
            saved.append({"name": f.name, "path": str(dest), "size": len(f.getvalue())})
        st.session_state.image_files = saved

    files = st.session_state.image_files
    if not files:
        st.info("No images uploaded yet. Use the file uploader above.")
        return

    st.markdown(f"**{len(files)} image(s) uploaded**")

    n_cols = min(len(files), 4)
    cols = st.columns(n_cols)
    for i, f in enumerate(files):
        with cols[i % n_cols]:
            st.image(f["path"], caption=f["name"], use_container_width=True)
            img = Image.open(f["path"])
            st.caption(f"{img.size[0]}x{img.size[1]} | {f['size'] / 1024:.0f} KB")

    st.markdown("")
    col_l, col_r = st.columns([1, 1])
    with col_r:
        if st.button("Next: Configure Pipelines >>", type="primary", use_container_width=True):
            _go_to("configure")
            st.rerun()


# ---------------------------------------------------------------------------
# Step 2: Configure
# ---------------------------------------------------------------------------

def _page_configure():
    if not st.session_state.image_files:
        st.warning("No images uploaded. Go back to Step 1.")
        if st.button("<< Back to Upload"):
            _go_to("upload")
            st.rerun()
        return

    st.subheader("Select Pipelines")

    available = get_available_pipelines()
    pipelines_info = list_pipelines()

    categories: dict[str, list[dict]] = {}
    for p in pipelines_info:
        categories.setdefault(p["category"], []).append(p)

    col_buttons, _ = st.columns([2, 1])
    with col_buttons:
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Select All", use_container_width=True):
                for p in pipelines_info:
                    st.session_state[f"pipe_{p['key']}"] = True
        with b2:
            if st.button("Select None", use_container_width=True):
                for p in pipelines_info:
                    st.session_state[f"pipe_{p['key']}"] = False
        with b3:
            if st.button("Select Available Only", use_container_width=True):
                for p in pipelines_info:
                    st.session_state[f"pipe_{p['key']}"] = p["key"] in available

    st.markdown("")

    selected_keys = []
    for cat_name, cat_pipelines in categories.items():
        color = CATEGORY_COLORS.get(cat_name, (128, 128, 128))
        r, g, b = color

        st.markdown(
            f'<div style="background:rgb({r},{g},{b});color:white;padding:6px 12px;'
            f'border-radius:6px;margin:8px 0 4px 0;font-weight:bold">{cat_name}</div>',
            unsafe_allow_html=True,
        )

        cat_cols = st.columns(len(cat_pipelines))
        for j, p in enumerate(cat_pipelines):
            with cat_cols[j]:
                is_available = p["key"] in available
                if f"pipe_{p['key']}" not in st.session_state:
                    st.session_state[f"pipe_{p['key']}"] = False

                checked = st.checkbox(
                    p["name"],
                    key=f"pipe_{p['key']}",
                    disabled=not is_available,
                    help=p["description"] if is_available else f"Missing credentials for {p['name']}",
                )

                if not is_available:
                    st.caption("Missing credentials")
                else:
                    st.caption(p["description"])

                if checked and is_available:
                    selected_keys.append(p["key"])

    st.session_state.selected_keys = selected_keys

    st.divider()

    st.session_state.run_parallel = st.checkbox(
        "Run pipelines in parallel",
        value=st.session_state.run_parallel,
    )

    st.markdown("")
    n_images = len(st.session_state.image_files)
    n_pipes = len(selected_keys)
    st.markdown(f"**{n_images} images x {n_pipes} pipelines = {n_images * n_pipes} total runs**")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        if st.button("<< Back to Upload", use_container_width=True):
            _go_to("upload")
            st.rerun()
    with col_r:
        if st.button(
            f"Run Comparison ({n_images * n_pipes} runs) >>",
            type="primary",
            use_container_width=True,
            disabled=n_pipes == 0,
        ):
            st.session_state.results = _run_all(
                st.session_state.image_files,
                selected_keys,
                st.session_state.run_parallel,
            )
            _go_to("results")
            st.rerun()


def _run_all(
    files: list[dict],
    pipeline_keys: list[str],
    parallel: bool,
) -> dict[str, list[PipelineResult]]:
    all_results = {}
    total = len(files) * len(pipeline_keys)
    progress = st.progress(0, text="Starting...")
    done = 0

    for f in files:
        image_path = f["path"]
        image_name = f["name"]

        results = []
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for key in pipeline_keys:
                    pipeline = get_pipeline(key)
                    futures[executor.submit(pipeline.execute, image_path)] = key

                for future in concurrent.futures.as_completed(futures):
                    done += 1
                    try:
                        result = future.result()
                    except Exception as e:
                        key = futures[future]
                        result = PipelineResult(
                            pipeline_name=key,
                            category="Error",
                            error=str(e),
                        )
                    results.append(result)
                    progress.progress(
                        done / total,
                        text=f"{image_name}: {done}/{total} completed",
                    )
        else:
            for key in pipeline_keys:
                pipeline = get_pipeline(key)
                try:
                    result = pipeline.execute(image_path)
                except Exception as e:
                    result = PipelineResult(
                        pipeline_name=key,
                        category="Error",
                        error=str(e),
                    )
                results.append(result)
                done += 1
                progress.progress(
                    done / total,
                    text=f"{image_name}: {done}/{total} completed",
                )

        all_results[image_name] = results

    progress.empty()
    return all_results


# ---------------------------------------------------------------------------
# Step 3: Results
# ---------------------------------------------------------------------------

def _page_results():
    results = st.session_state.results

    if not results:
        st.warning("No results yet. Run a comparison first.")
        if st.button("<< Back to Configure"):
            _go_to("configure")
            st.rerun()
        return

    st.subheader("Comparison Results")

    image_names = list(results.keys())

    view_mode = st.radio(
        "View mode",
        ["Per Image", "Cross-Image Summary"],
        horizontal=True,
    )

    if view_mode == "Per Image":
        if len(image_names) > 1:
            tabs = st.tabs(image_names)
        else:
            tabs = [st.container()]

        for idx, image_name in enumerate(image_names):
            with tabs[idx]:
                _render_image_results(image_name, results[image_name])

    else:
        _render_cross_image_summary(results)

    st.divider()

    with st.expander("Save Results"):
        save_label = st.text_input(
            "Label (optional)",
            placeholder="e.g. outdoor-test-batch-1",
            key="save_label",
        )
        if st.button("Save", type="primary", use_container_width=True):
            filepath = _save_results(
                save_label,
                st.session_state.results,
                st.session_state.image_files,
            )
            st.success(f"Saved to {filepath.name}")

    col_l, col_r = st.columns([1, 1])
    with col_l:
        if st.button("<< Back to Configure", use_container_width=True):
            _go_to("configure")
            st.rerun()
    with col_r:
        if st.button("New Comparison (reset)", use_container_width=True):
            st.session_state.results = {}
            st.session_state.image_files = []
            _go_to("upload")
            st.rerun()


def _render_image_results(image_name: str, image_results: list[PipelineResult]):
    file_info = next(
        (f for f in st.session_state.image_files if f["name"] == image_name),
        None,
    )
    image_path = file_info["path"] if file_info else None

    if image_path:
        with st.expander("Original Image", expanded=False):
            st.image(image_path, caption=image_name, width=400)

    n_cols = min(len(image_results), 3)
    cols = st.columns(n_cols)

    for i, result in enumerate(image_results):
        with cols[i % n_cols]:
            cat_color = CATEGORY_COLORS.get(result.category, (128, 128, 128))
            r, g, b = cat_color
            st.markdown(
                f'<div style="background:rgb({r},{g},{b});color:white;padding:4px 8px;'
                f'border-radius:4px;font-size:0.8em;margin-bottom:4px">{result.category}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{result.pipeline_name}**")

            if result.error:
                st.error(f"Error: {result.error}")
                continue

            if image_path:
                annotated = draw_detections(image_path, result.boxes, result.pipeline_name)
                st.image(annotated, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Latency", f"{result.latency_ms:.0f}ms")
            m2.metric("$/1K", f"${result.cost_per_1000:.2f}")
            m3.metric("Found", result.box_count)

            with st.expander("Details"):
                for box in result.boxes:
                    if box.x_max <= 0:
                        st.text(f"  {box.label} ({box.confidence:.0f}%) - no bbox")
                        continue
                    parts = [box.label]
                    if box.brand:
                        parts.append(f"Brand: {box.brand}")
                    if box.product_name:
                        parts.append(f"Product: {box.product_name}")
                    parts.append(f"{box.confidence:.0f}%")
                    st.text("  " + " | ".join(parts))

    st.divider()
    _render_comparison_table(image_results, image_name)


def _render_comparison_table(results: list[PipelineResult], title: str = ""):
    valid = [r for r in results if not r.error]
    if not valid:
        return

    rows = []
    for r in valid:
        rows.append({
            "Pipeline": r.pipeline_name,
            "Category": r.category,
            "Latency (ms)": round(r.latency_ms),
            "Cost/1K ($)": round(r.cost_per_1000, 2),
            "Detections": r.box_count,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    fastest = min(valid, key=lambda r: r.latency_ms)
    cheapest = min(valid, key=lambda r: r.cost_estimate)
    most = max(valid, key=lambda r: r.box_count)

    c1, c2, c3 = st.columns(3)
    c1.success(f"Fastest: **{fastest.pipeline_name}** ({fastest.latency_ms:.0f}ms)")
    c2.success(f"Cheapest: **{cheapest.pipeline_name}** (${cheapest.cost_per_1000:.2f}/1K)")
    c3.success(f"Most Detections: **{most.pipeline_name}** ({most.box_count})")


def _render_cross_image_summary(all_results: dict[str, list[PipelineResult]]):
    st.markdown("### Aggregated Results Across All Images")

    pipeline_stats: dict[str, dict] = {}

    for image_name, results in all_results.items():
        for r in results:
            if r.error:
                continue
            if r.pipeline_name not in pipeline_stats:
                pipeline_stats[r.pipeline_name] = {
                    "category": r.category,
                    "total_latency": 0,
                    "total_cost": 0,
                    "total_detections": 0,
                    "image_count": 0,
                    "errors": 0,
                }
            stats = pipeline_stats[r.pipeline_name]
            stats["total_latency"] += r.latency_ms
            stats["total_cost"] += r.cost_estimate
            stats["total_detections"] += r.box_count
            stats["image_count"] += 1

    if not pipeline_stats:
        st.warning("No successful results to aggregate.")
        return

    rows = []
    for name, stats in pipeline_stats.items():
        n = stats["image_count"]
        rows.append({
            "Pipeline": name,
            "Category": stats["category"],
            "Avg Latency (ms)": round(stats["total_latency"] / n),
            "Avg Cost/1K ($)": round(stats["total_cost"] / n * 1000, 2),
            "Avg Detections": round(stats["total_detections"] / n, 1),
            "Images Processed": n,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    best_latency = min(rows, key=lambda r: r["Avg Latency (ms)"])
    best_cost = min(rows, key=lambda r: r["Avg Cost/1K ($)"])
    best_detect = max(rows, key=lambda r: r["Avg Detections"])

    c1, c2, c3 = st.columns(3)
    c1.success(f"Fastest Avg: **{best_latency['Pipeline']}** ({best_latency['Avg Latency (ms)']}ms)")
    c2.success(f"Cheapest Avg: **{best_cost['Pipeline']}** (${best_cost['Avg Cost/1K ($)']}/1K)")
    c3.success(f"Most Detections Avg: **{best_detect['Pipeline']}** ({best_detect['Avg Detections']})")

    st.markdown("### Per-Image Breakdown")
    for image_name, results in all_results.items():
        with st.expander(f"{image_name} ({len([r for r in results if not r.error])} results)"):
            _render_comparison_table(results, image_name)


if __name__ == "__main__":
    main()
