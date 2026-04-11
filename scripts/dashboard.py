"""
dashboard.py - Interactive web dashboard for hotspot detection.

A Streamlit application for uploading images, running detection,
visualizing results, and downloading outputs.

Run with:
    streamlit run scripts/dashboard.py
"""

import datetime
import io
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Configuration
API_BASE_URL = "http://127.0.0.1:9000"
DETECT_ENDPOINT = f"{API_BASE_URL}/detect"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "outputs" / "training"
MODEL_DIR = PROJECT_ROOT / "models"


def initialize_session():
    """Initialize Streamlit session state."""
    if "detection_result" not in st.session_state:
        st.session_state.detection_result = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    if "api_available" not in st.session_state:
        st.session_state.api_available = False


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def run_detection(image_bytes, confidence_threshold=0.5):
    """
    Run detection on image bytes via API.

    Args:
        image_bytes: Image file bytes
        confidence_threshold: Confidence threshold (0.0-1.0)

    Returns:
        Detection response or None if error
    """
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        params = {"confidence_threshold": confidence_threshold}

        response = requests.post(DETECT_ENDPOINT, files=files, params=params, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}\n{response.text}")
            return None

    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Is the server running?")
        return None
    except Exception as e:
        st.error(f"Error running detection: {e}")
        return None


def draw_detections_pil(image_pil, detections):
    """
    Draw detection boxes and labels on image using PIL.

    Args:
        image_pil: PIL Image object
        detections: List of detection dicts

    Returns:
        PIL Image with drawn detections
    """
    img_copy = image_pil.copy()
    draw = ImageDraw.Draw(img_copy)

    # Colors for different classes
    colors = {
        "weed": "#FF0000",
        "disease": "#FF8800",
        "pest": "#8800FF",
        "anomaly": "#FFFF00",
        "unknown": "#FFFFFF",
    }

    for i, det in enumerate(detections):
        bbox = det.get("bbox", {})
        x1 = bbox.get("x1", 0)
        y1 = bbox.get("y1", 0)
        x2 = bbox.get("x2", 0)
        y2 = bbox.get("y2", 0)

        class_name = det.get("class_name", "unknown")
        confidence = det.get("confidence", 0)

        # Get color for class
        color = colors.get(class_name, "#FFFFFF")

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"{class_name} {confidence:.2f}"
        try:
            # Draw text background
            bbox_text = draw.textbbox((x1, y1 - 25), label)
            draw.rectangle(bbox_text, fill=color)
            draw.text((x1, y1 - 25), label, fill="#000000")
        except Exception:
            # Fallback if font issue
            draw.text((x1, y1 - 20), label, fill=color)

        # Draw polygon if available
        polygon = det.get("polygon")
        if polygon:
            try:
                points = [(p[0], p[1]) for p in polygon]
                if len(points) > 1:
                    draw.polygon(points, outline=color, width=2)
            except Exception:
                pass

    return img_copy


def render_detection_table(detections):
    """Render detection results as DataFrame."""
    data = []
    for i, det in enumerate(detections):
        bbox = det.get("bbox", {})
        data.append({
            "ID": det.get("id", f"det_{i}"),
            "Class": det.get("class_name", "unknown"),
            "Confidence": f"{det.get('confidence', 0):.3f}",
            "Severity": f"{det.get('severity_score', 0):.3f}" if det.get("severity_score") else "N/A",
            "Area (px)": f"{bbox.get('area', 0):.0f}",
            "Width": f"{bbox.get('width', 0):.0f}",
            "Height": f"{bbox.get('height', 0):.0f}",
        })

    return pd.DataFrame(data)


def export_detections_csv(detections, source_image):
    """Export detections to CSV format."""
    data = []
    for i, det in enumerate(detections):
        bbox = det.get("bbox", {})
        data.append({
            "detection_id": det.get("id", f"det_{i}"),
            "class_name": det.get("class_name", "unknown"),
            "confidence": det.get("confidence", 0),
            "severity_score": det.get("severity_score"),
            "bbox_x1": bbox.get("x1", 0),
            "bbox_y1": bbox.get("y1", 0),
            "bbox_x2": bbox.get("x2", 0),
            "bbox_y2": bbox.get("y2", 0),
            "bbox_width": bbox.get("width", 0),
            "bbox_height": bbox.get("height", 0),
            "bbox_area": bbox.get("area", 0),
            "source_image": source_image,
            "timestamp": det.get("timestamp"),
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


# ═══════════════════════════════════════════════════════════════════
#  Training Monitoring Helpers
# ═══════════════════════════════════════════════════════════════════

def discover_runs():
    """Return list of training run names that have a results.csv."""
    if not TRAINING_DIR.exists():
        return []
    return sorted(
        d.name for d in TRAINING_DIR.iterdir()
        if d.is_dir() and (d / "results.csv").exists()
    )


def load_results_csv(run_name):
    """Load and clean a run's results.csv into a DataFrame."""
    csv_path = TRAINING_DIR / run_name / "results.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def get_run_status(run_name):
    """Determine if a run is completed or still in progress."""
    run_dir = TRAINING_DIR / run_name
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if best.exists():
        return "Completed"
    if last.exists():
        return "Training (in progress)"
    return "Unknown"


def get_model_info(run_name):
    """Return dict with model file metadata."""
    best = TRAINING_DIR / run_name / "weights" / "best.pt"
    deployed = MODEL_DIR / "yolo_crop_disease.pt"
    info = {}
    for label, path in [("best.pt", best), ("deployed model", deployed)]:
        if path.exists():
            stat = path.stat()
            info[label] = {
                "path": str(path),
                "size_mb": stat.st_size / 1e6,
                "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
    return info


def render_metrics_panel(df):
    """Render the latest training metrics as st.metric cards."""
    if df is None or df.empty:
        st.warning("No training data available.")
        return

    last = df.iloc[-1]
    best_idx = df["metrics/mAP50(B)"].idxmax()
    best = df.loc[best_idx]
    prev = df.iloc[-2] if len(df) >= 2 else last

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Epoch",
            f"{int(last['epoch'])} / {int(df['epoch'].max())}",
        )
    with c2:
        delta = float(last["metrics/mAP50(B)"] - prev["metrics/mAP50(B)"])
        st.metric("mAP@0.5", f"{last['metrics/mAP50(B)']:.4f}", f"{delta:+.4f}")
    with c3:
        delta = float(last["metrics/precision(B)"] - prev["metrics/precision(B)"])
        st.metric("Precision", f"{last['metrics/precision(B)']:.4f}", f"{delta:+.4f}")
    with c4:
        delta = float(last["metrics/recall(B)"] - prev["metrics/recall(B)"])
        st.metric("Recall", f"{last['metrics/recall(B)']:.4f}", f"{delta:+.4f}")

    # Best epoch highlight
    st.caption(
        f"Best mAP@0.5: **{best['metrics/mAP50(B)']:.4f}** at epoch "
        f"**{int(best['epoch'])}**"
    )


def render_training_charts(df):
    """Render mAP and loss curves."""
    if df is None or df.empty:
        return

    tab_map, tab_loss = st.tabs(["Detection Metrics", "Loss Curves"])

    with tab_map:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], "b-o", ms=3, label="mAP@0.5")
        ax.plot(df["epoch"], df["metrics/mAP50-95(B)"], "r-s", ms=3, label="mAP@0.5:0.95")
        ax.plot(df["epoch"], df["metrics/precision(B)"], "g--", alpha=0.6, label="Precision")
        ax.plot(df["epoch"], df["metrics/recall(B)"], color="orange", linestyle="--", alpha=0.6, label="Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Detection Metrics per Epoch")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tab_loss:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, col, title in zip(
            axes,
            ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
            ["Box Loss", "Classification Loss", "DFL Loss"],
        ):
            ax.plot(df["epoch"], df[col], label="train")
            val_col = col.replace("train/", "val/")
            if val_col in df.columns:
                ax.plot(df["epoch"], df[val_col], "--", alpha=0.7, label="val")
            ax.set_xlabel("Epoch")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_model_info(run_name):
    """Render model file information panel."""
    info = get_model_info(run_name)
    if not info:
        st.info("No model weights found for this run.")
        return

    for label, meta in info.items():
        st.markdown(f"**{label}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.code(meta["path"], language=None)
        with c2:
            st.metric("Size", f"{meta['size_mb']:.2f} MB")
        with c3:
            st.metric("Modified", meta["modified"])


def main():
    """Main Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="AgriDrone Detection",
        page_icon="🚜",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    initialize_session()

    # Title and description
    st.title("🚜 AgriDrone Hotspot Detection")
    st.markdown(
        """
        **Research Prototype for Site-Specific Crop Protection**

        Upload field imagery to detect crop stress hotspots (weeds, disease, pests).
        Results include bounding boxes, confidence scores, and exportable maps.
        """
    )

    # Check API health
    st.session_state.api_available = check_api_health()

    # Sidebar
    st.sidebar.header("⚙️ Settings")

    if not st.session_state.api_available:
        st.sidebar.warning(
            "⚠️ **API Unavailable**\n\n"
            "Make sure the FastAPI server is running:\n"
            "```bash\n"
            "python -m uvicorn agridrone.api.app:app --port 9000\n"
            "```"
        )

    # ── Run Selector (sidebar) ──
    available_runs = discover_runs()
    selected_run = None
    if available_runs:
        st.sidebar.divider()
        st.sidebar.header("📂 Training Run")
        selected_run = st.sidebar.selectbox(
            "Select run", available_runs,
            index=len(available_runs) - 1,
        )
        run_status = get_run_status(selected_run)
        if run_status == "Completed":
            st.sidebar.success(f"🟢 {run_status}")
        else:
            st.sidebar.warning(f"🟡 {run_status}")

    st.sidebar.divider()

    # ── Confidence slider (used for both detection & prediction) ──
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections & predictions",
    )

    st.sidebar.divider()

    # File uploader
    st.sidebar.header("📤 Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Select an image file",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload field imagery (JPG, PNG, BMP)",
    )

    # Main content area
    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file

        # Create columns for layout
        col_upload, col_results = st.columns([1, 1])

        # Left column: Image upload and preview
        with col_upload:
            st.subheader("📷 Original Image")

            image_bytes = uploaded_file.read()
            image_pil = Image.open(io.BytesIO(image_bytes))

            st.image(image_pil, use_container_width=True)

            # Run detection button
            if st.session_state.api_available:
                run_button = st.button(
                    "🔍 Run Detection",
                    key="run_detection",
                    type="primary",
                    use_container_width=True,
                )

                if run_button:
                    with st.spinner("Running detection... this may take a moment"):
                        st.session_state.detection_result = run_detection(
                            image_bytes, confidence_threshold
                        )

            else:
                st.error("API not available. Cannot run detection.")

        # Right column: Results
        with col_results:
            if st.session_state.detection_result is not None:
                result = st.session_state.detection_result

                if result.get("status") == "success":
                    detections = result.get("detections", [])
                    num_detections = result.get("num_detections", 0)
                    processing_time = result.get("processing_time_ms", 0)

                    # Draw detections
                    st.subheader("🎯 Detection Results")

                    image_with_boxes = draw_detections_pil(image_pil, detections)
                    st.image(image_with_boxes, use_container_width=True)

                    # Statistics
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Detections Found", num_detections)
                    with col_stat2:
                        st.metric("Processing Time", f"{processing_time:.1f}ms")

                else:
                    st.error(f"Detection failed: {result.get('error', 'Unknown error')}")

            else:
                st.info("👈 Upload an image and click 'Run Detection' to see results")

    else:
        st.info("👈 Start by uploading an image in the sidebar")

    # Results section
    if st.session_state.detection_result is not None and st.session_state.detection_result.get("status") == "success":
        result = st.session_state.detection_result
        detections = result.get("detections", [])

        st.divider()
        st.subheader("📊 Detection Details")

        # Detection table
        if detections:
            df = render_detection_table(detections)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download section
            st.subheader("💾 Export Results")

            col_csv, col_json = st.columns(2)

            # CSV download
            with col_csv:
                csv_data = export_detections_csv(
                    detections,
                    st.session_state.uploaded_image.name
                )
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # JSON download
            with col_json:
                json_data = json.dumps(result, indent=2, default=str)
                st.download_button(
                    label="📥 Download as JSON",
                    data=json_data,
                    file_name="detections.json",
                    mime="application/json",
                    use_container_width=True,
                )

        else:
            st.info("No detections found in this image.")

        # Raw result viewer
        st.divider()
        st.subheader("🔍 Raw Response")

        with st.expander("View full API response"):
            st.json(result)

    # ═══════════════════════════════════════════════════════════════
    #  Training Monitoring & Control Section
    # ═══════════════════════════════════════════════════════════════
    if selected_run:
        st.divider()
        st.header("📊 Training Monitor")

        # Refresh button
        col_title, col_btn = st.columns([4, 1])
        with col_title:
            st.caption(f"Run: **{selected_run}**  ·  Status: **{get_run_status(selected_run)}**")
        with col_btn:
            refresh = st.button("🔄 Refresh Metrics", use_container_width=True)

        # Load CSV (re-read on every render or refresh click)
        df_results = load_results_csv(selected_run)

        # ── Metrics Cards ──
        with st.container():
            st.subheader("Metrics")
            render_metrics_panel(df_results)

        # ── Charts ──
        with st.container():
            st.subheader("📈 Charts")
            render_training_charts(df_results)

        # ── Model Info ──
        with st.container():
            st.subheader("📁 Model Info")
            render_model_info(selected_run)

        # ── Quick Prediction ──
        with st.container():
            st.subheader("🧪 Quick Prediction")
            st.caption(f"Confidence threshold: **{confidence_threshold}**")
            pred_file = st.file_uploader(
                "Upload image for YOLO prediction",
                type=["jpg", "jpeg", "png"],
                key="pred_upload",
            )
            if pred_file is not None:
                best_pt = TRAINING_DIR / selected_run / "weights" / "best.pt"
                deployed_pt = MODEL_DIR / "yolo_crop_disease.pt"
                model_path = best_pt if best_pt.exists() else deployed_pt

                if not model_path.exists():
                    st.error("No model weights found. Train a model first.")
                else:
                    pred_bytes = pred_file.read()
                    pred_img = Image.open(io.BytesIO(pred_bytes))
                    col_orig, col_pred = st.columns(2)
                    with col_orig:
                        st.image(pred_img, caption="Input", use_container_width=True)

                    try:
                        from ultralytics import YOLO
                        yolo = YOLO(str(model_path))
                        results_pred = yolo.predict(
                            source=np.array(pred_img),
                            imgsz=640,
                            conf=confidence_threshold,
                            device="cpu",
                            verbose=False,
                        )
                        with col_pred:
                            annotated = results_pred[0].plot()
                            st.image(
                                annotated[:, :, ::-1],
                                caption=f"{len(results_pred[0].boxes)} detections",
                                use_container_width=True,
                            )
                        # Detection details
                        if len(results_pred[0].boxes):
                            det_data = []
                            for box in results_pred[0].boxes:
                                cls_id = int(box.cls[0])
                                det_data.append({
                                    "Class": yolo.names[cls_id],
                                    "Confidence": f"{float(box.conf[0]):.3f}",
                                    "Box": f"{box.xyxy[0].tolist()}",
                                })
                            st.dataframe(pd.DataFrame(det_data), hide_index=True)
                        else:
                            st.info("No detections at this confidence level.")
                    except ImportError:
                        st.error("ultralytics not installed. Run: pip install ultralytics")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")


if __name__ == "__main__":
    main()
