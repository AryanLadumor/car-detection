import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Traffic Region Editor", layout="wide")
st.title("YOLOv8 Traffic Density Estimation")
st.subheader("Video Upload & Quadrilateral Region Editor")
st.write("Upload a video, choose the preview size, adjust the lane corners, then run YOLOv8 vehicle detection when ready.")

# Original regions are retained as the default for the first two lanes.
DEFAULT_VERTICES = {
    "Lane 1": [(465, 350), (609, 350), (510, 630), (2, 630)],
    "Lane 2": [(678, 350), (815, 350), (1203, 630), (743, 630)],
}
MAX_LANES = 6


def make_default_lane(lane_index, lane_count, frame_width, frame_height):
    """Create a reasonable trapezoid placeholder for additional lanes."""
    # Perspective-friendly trapezoids: narrower near the horizon, wider at the bottom.
    top_y = int(frame_height * (350 / 720))
    bottom_y = int(frame_height * (630 / 720))
    top_left = int(frame_width * (lane_index / lane_count))
    top_right = int(frame_width * ((lane_index + 1) / lane_count))
    bottom_left = int(frame_width * (lane_index / lane_count))
    bottom_right = int(frame_width * ((lane_index + 1) / lane_count))
    return [(top_left, top_y), (top_right, top_y), (bottom_right, bottom_y), (bottom_left, bottom_y)]


uploaded_video = st.file_uploader(
    "Upload a traffic video", type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_video is None:
    st.info("Upload a video to start. The default quadrilaterals are taken from your original project.")
    st.stop()

# Keep the upload in a temporary file with the original extension so OpenCV can decode it.
suffix = Path(uploaded_video.name).suffix or ".mp4"
video_bytes = uploaded_video.getvalue()
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
    temp_video.write(video_bytes)
    video_path = temp_video.name

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    st.error("OpenCV could not open this video. Try an MP4 or AVI file.")
    cap.release()
    st.stop()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if frame_count <= 0 or width <= 0 or height <= 0:
    st.error("Could not read valid video metadata.")
    cap.release()
    st.stop()

st.caption(f"File: {uploaded_video.name} · Resolution: {width} × {height} · Frames: {frame_count} · FPS: {fps:.2f}")

# Read a selectable frame for preview.
frame_index = st.slider("Preview frame", min_value=0, max_value=frame_count - 1, value=0, step=1)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ret, frame = cap.read()
cap.release()
if not ret:
    st.error("Could not decode the selected frame.")
    st.stop()

# Initialize session coordinates once per uploaded video. Coordinates use original-video pixels.
video_key = f"{uploaded_video.name}:{len(video_bytes)}:{width}x{height}"
if st.session_state.get("video_key") != video_key:
    st.session_state["video_key"] = video_key
    sx, sy = width / 1280, height / 720
    st.session_state["regions"] = {
        name: [(round(x * sx), round(y * sy)) for x, y in points]
        for name, points in DEFAULT_VERTICES.items()
    }
    st.session_state["lane_count"] = 2
    st.session_state.pop("processed_video_bytes", None)

# Lane count selector (1–6). Existing edited lanes are preserved when possible.
selected_lane_count = st.selectbox(
    "Number of lanes / road regions to configure",
    options=list(range(1, MAX_LANES + 1)),
    index=min(max(int(st.session_state.get("lane_count", 2)), 1), MAX_LANES) - 1,
    help="Each lane is a four-corner polygon. Drag its four handles on the video preview.",
    key=f"lane-count-{video_key}",
)
regions = st.session_state["regions"]
old_count = len(regions)
if selected_lane_count != old_count:
    resized_regions = {}
    for i in range(selected_lane_count):
        name = f"Lane {i + 1}"
        if name in regions:
            resized_regions[name] = regions[name]
        elif i < 2:
            sx, sy = width / 1280, height / 720
            resized_regions[name] = [(round(x * sx), round(y * sy)) for x, y in DEFAULT_VERTICES[name]]
        else:
            resized_regions[name] = make_default_lane(i, selected_lane_count, width, height)
    st.session_state["regions"] = resized_regions
    st.session_state["lane_count"] = selected_lane_count
    st.rerun()

regions = st.session_state["regions"]

# Preview size is independent of detection. Changing it reruns only the Streamlit UI;
# it does not start or rerun YOLO inference.
preview_size = st.selectbox(
    "Video preview size",
    options=["Small", "Medium", "Large"],
    index=1,
    key="video_preview_size",
    help="Changes the size of the editable video preview immediately. Detection starts only when you click Run detection.",
)
preview_widths = {"Small": 640, "Medium": 1000, "Large": 1400}
preview_width = preview_widths[preview_size]
canvas_width = max(1, min(width, preview_widths[preview_size]))
canvas_height = max(1, round(height * canvas_width / width))
scale_x = width / canvas_width
scale_y = height / canvas_height

# Distinct, high-contrast colors in OpenCV BGR and canvas RGBA formats.
LANE_BGR = [(0, 255, 0), (0, 150, 255), (255, 80, 0), (255, 0, 200), (0, 220, 220), (180, 80, 255)]
LANE_RGBA = ["rgba(0,220,0,0.95)", "rgba(255,150,0,0.95)", "rgba(0,100,255,0.95)", "rgba(230,0,180,0.95)", "rgba(0,200,200,0.95)", "rgba(160,70,255,0.95)"]

controls_col, preview_col = st.columns([1, 1.8], gap="large")
with controls_col:
    st.markdown("### Edit lane corner points")
    st.write("Drag the numbered handles on the video. Each lane is a quadrilateral with four draggable corners.")
    if st.button("Reset lane regions", width='stretch'):
        sx, sy = width / 1280, height / 720
        reset_regions = {}
        for i in range(selected_lane_count):
            name = f"Lane {i + 1}"
            if i < 2:
                reset_regions[name] = [(round(x * sx), round(y * sy)) for x, y in DEFAULT_VERTICES[name]]
            else:
                reset_regions[name] = make_default_lane(i, selected_lane_count, width, height)
        st.session_state["regions"] = reset_regions
        st.session_state.pop("processed_video_bytes", None)

        # Force streamlit-drawable-canvas to rebuild from the reset coordinates.
        # Without a new component key, its browser-side drawing can keep the old
        # points and overwrite the reset regions on the next rerun.
        st.session_state["region_reset_version"] = (
            st.session_state.get("region_reset_version", 0) + 1
        )
        st.rerun()
    for lane_index, (lane_name, points) in enumerate(regions.items()):
        st.markdown(f"**Lane {lane_index + 1}**")
        st.caption(" · ".join(f"P{i + 1}: ({x}, {y})" for i, (x, y) in enumerate(points)))

with preview_col:
    st.markdown("### Drag points on the video")
    preview = frame.copy()
    for lane_index, (lane_name, points_list) in enumerate(regions.items()):
        color = LANE_BGR[lane_index]
        points = np.array(points_list, dtype=np.int32)
        cv2.polylines(preview, [points], isClosed=True, color=color, thickness=max(2, round(width / 640)))
        for point_index, (px, py) in enumerate(points, start=1):
            cv2.putText(preview, f"{lane_index + 1}.{point_index}", (int(px) + 8, int(py) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.45, width / 1800), color, 2, cv2.LINE_AA)
    if canvas_width != width:
        preview = cv2.resize(preview, (canvas_width, canvas_height))
    background_image = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

    radius = max(7, round(canvas_width / 100))
    canvas_objects = []
    # Stable ordering: each lane's four points, sequentially.
    for lane_index, (lane_name, points) in enumerate(regions.items()):
        for point_index, (px, py) in enumerate(points, start=1):
            cx, cy = px / scale_x, py / scale_y
            canvas_objects.append({
                "type": "circle", "left": cx - radius, "top": cy - radius,
                "originX": "left", "originY": "top", "radius": radius,
                "fill": LANE_RGBA[lane_index], "stroke": "white", "strokeWidth": 2,
                "selectable": True, "evented": True, "hasControls": False,
                "hasBorders": True, "objectCaching": False,
                "name": f"{lane_name} point {point_index}",
            })

    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0.1)",
        stroke_width=1,
        background_image=background_image,
        initial_drawing={"version": "5.3.0", "objects": canvas_objects},
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="point",  # Avoid creating a new point/dot on every canvas click; use the Edit toggle to drag existing handles.
        key=(
            f"drag-canvas-{video_key}-{frame_index}-{selected_lane_count}-"
            f"{canvas_width}-{st.session_state.get('region_reset_version', 0)}"
        ),
    )

    if canvas_result.json_data is not None:
        objects = canvas_result.json_data.get("objects", [])
        expected_objects = selected_lane_count * 4

        # The canvas library still allows drawing in its selected drawing mode.
        # If a click creates an extra shape, immediately discard it by rebuilding
        # the canvas from the saved lane handles. Only the original lane handles
        # are allowed to remain on the preview.
        if len(objects) > expected_objects:
            st.rerun()

        if len(objects) >= expected_objects:
            updated_regions = {}
            for lane_index in range(selected_lane_count):
                lane_name = f"Lane {lane_index + 1}"
                lane_points = []
                for obj in objects[lane_index * 4:(lane_index + 1) * 4]:
                    radius_value = float(obj.get("radius", radius))
                    center_x = float(obj.get("left", 0)) + radius_value
                    center_y = float(obj.get("top", 0)) + radius_value
                    original_x = int(round(center_x * scale_x))
                    original_y = int(round(center_y * scale_y))
                    lane_points.append((min(max(original_x, 0), width - 1), min(max(original_y, 0), height - 1)))
                updated_regions[lane_name] = lane_points
            if updated_regions != st.session_state["regions"]:
                st.session_state["regions"] = updated_regions
                st.session_state.pop("processed_video_bytes", None)
                st.rerun()
    st.markdown("**Legend:** Each color represents a different lane. Handles are labeled lane.point (for example, 2.3 = Lane 2, point 3).")

# Export all configured lanes while retaining vertices1/vertices2 for older integrations.
export_data = {
    "video": uploaded_video.name,
    "frame_width": width,
    "frame_height": height,
    "lane_count": len(st.session_state["regions"]),
    "lanes": {name: [list(point) for point in points] for name, points in st.session_state["regions"].items()},
}
for i, points in enumerate(st.session_state["regions"].values(), start=1):
    export_data[f"vertices{i}"] = [list(point) for point in points]

st.divider()
st.header("Run vehicle detection")
st.write(
    "Run the YOLOv8 model on the uploaded video. Detected vehicles are counted "
    "inside each edited quadrilateral; the output video includes detection boxes, "
    "region outlines, counts, and traffic-intensity labels."
)

model_path = Path(__file__).resolve().parent / "models" / "best.pt"

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo_model(path_string):
    from ultralytics import YOLO
    return YOLO(path_string)

run_col, conf_col, threshold_col = st.columns([1.4, 1, 1])
with conf_col:
    confidence = st.slider("Detection confidence", 0.05, 0.95, 0.40, 0.05)
with threshold_col:
    heavy_threshold = st.number_input("Heavy traffic threshold", min_value=0, max_value=500, value=10, step=1)

with run_col:
    st.caption("The model is loaded only when you start analysis.")

if st.button("Run detection on uploaded video", type="primary", use_container_width=True):
    if not model_path.exists():
        st.error(f"Model not found: {model_path}. Place your fine-tuned best.pt file in the project's models/ folder.")
        st.stop()

    # Snapshot every configured lane before processing.
    lane_polygons = {
        name: np.array(points, dtype=np.int32)
        for name, points in st.session_state["regions"].items()
    }
    # Cumulative unique vehicle counts, based on tracker IDs across the video.
    lane_counts_total = {name: 0 for name in lane_polygons}
    seen_vehicle_ids_by_lane = {name: set() for name in lane_polygons}
    seen_vehicle_ids_any_lane = set()

    try:
        model = load_yolo_model(str(model_path))
    except Exception as exc:
        st.error(f"Could not load the YOLO model: {exc}")
        st.stop()

    # A temporary output path avoids overwriting project files.
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_temp.name
    output_temp.close()

    processing_cap = cv2.VideoCapture(video_path)
    if not processing_cap.isOpened():
        st.error("Could not reopen the uploaded video for processing.")
        st.stop()

    video_fps = processing_cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0 or video_fps > 240:
        video_fps = 20.0

    output_width = int(processing_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(processing_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(processing_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MP4V is broadly supported by OpenCV on Windows. Check writer before processing.
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        processing_cap.release()
        st.error("Could not create the output video. Check that your OpenCV installation supports the mp4v codec.")
        st.stop()

    progress = st.progress(0, text="Starting vehicle detection...")
    st.caption("Live preview while processing. Counts are shown beside the video, not over it.")
    preview_col, counts_col = st.columns([3, 1], gap="medium")
    with preview_col:
        live_preview = st.empty()
    with counts_col:
        st.markdown("#### Live counts")
        live_counts_panel = st.empty()
        st.markdown("#### Cumulative passed")
        cumulative_counts_panel = st.empty()
    status = st.empty()
    frames_processed = 0

    # Resolve model class names and use vehicle-like labels where the model exposes them.
    names = model.names
    if isinstance(names, dict):
        name_items = names.items()
    else:
        name_items = enumerate(names)
    vehicle_terms = ("car", "truck", "bus", "motorcycle", "motorbike", "van", "vehicle", "pickup", "lorry")
    vehicle_class_ids = {
        int(class_id)
        for class_id, class_name in name_items
        if any(term in str(class_name).lower() for term in vehicle_terms)
    }
    filter_vehicle_classes = bool(vehicle_class_ids)

    try:
        while True:
            ret_frame, current_frame = processing_cap.read()
            if not ret_frame:
                break

            # Track vehicles across frames so cumulative totals count each tracked vehicle once.
            result = model.track(
                current_frame,
                imgsz=640,
                conf=confidence,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            # Draw YOLO bounding boxes and class/confidence labels for all detected objects,
            # like the original OpenCV script. Lane polygons are used separately for counting.
            annotated = result.plot(line_width=2)

            frame_lane_counts = {name: 0 for name in lane_polygons}
            total_vehicles_in_lanes = 0

            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item()) if box.cls is not None else -1
                    if filter_vehicle_classes and class_id not in vehicle_class_ids:
                        continue

                    x_min, y_min, x_max, y_max = [int(v) for v in box.xyxy[0].tolist()]
                    center_x = int((x_min + x_max) / 2)
                    center_y = int((y_min + y_max) / 2)
                    confidence_value = float(box.conf[0].item()) if box.conf is not None else 0.0

                    # Live counts are vehicles currently inside each lane.
                    inside_lanes = []
                    for lane_name, polygon in lane_polygons.items():
                        if cv2.pointPolygonTest(polygon, (float(center_x), float(center_y)), False) >= 0:
                            frame_lane_counts[lane_name] += 1
                            inside_lanes.append(lane_name)

                    # Cumulative counts use tracker IDs so the same vehicle is not
                    # repeatedly counted in every frame it remains visible.
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].item())

                    if inside_lanes:
                        total_vehicles_in_lanes += 1
                        if track_id is not None:
                            seen_vehicle_ids_any_lane.add(track_id)
                            for lane_name in inside_lanes:
                                if track_id not in seen_vehicle_ids_by_lane[lane_name]:
                                    seen_vehicle_ids_by_lane[lane_name].add(track_id)
                                    lane_counts_total[lane_name] += 1

            for lane_index, (lane_name, polygon) in enumerate(lane_polygons.items()):
                color = LANE_BGR[lane_index]
                cv2.polylines(annotated, [polygon], True, color, 3)
                # Put a lane label near the polygon's first corner.
                label_x, label_y = map(int, polygon[0])
                cv2.putText(annotated, f"{lane_name}: {frame_lane_counts[lane_name]}",
                            (max(5, label_x), max(25, label_y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, color, 2, cv2.LINE_AA)

            # Counts are rendered in a separate Streamlit side panel, not over the video.
            # This keeps the detection preview unobstructed.

            writer.write(annotated)
            frames_processed += 1

            # Show a live annotated frame in the Streamlit page during inference.
            if frames_processed == 1 or frames_processed % 3 == 0:
                preview_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                live_preview.image(
                    preview_rgb,
                    caption=f"Live detection preview — frame {frames_processed}/{total_frames}",
                    use_container_width=True,
                )
                live_counts_panel.markdown(
                    f"**Inside lanes now: {total_vehicles_in_lanes}**\n\n"
                    + "\n\n".join(
                        f"**{name}:** {count} ({'Heavy' if count > heavy_threshold else 'Smooth'})"
                        for name, count in frame_lane_counts.items()
                    )
                )
                cumulative_counts_panel.markdown(
                    f"**Unique vehicles passed (all lanes): {len(seen_vehicle_ids_any_lane)}**\n\n"
                    + "\n\n".join(
                        f"**{name} passed:** {lane_counts_total[name]}"
                        for name in lane_polygons
                    )
                )

            if total_frames > 0:
                progress.progress(
                    min(frames_processed / total_frames, 1.0),
                    text=f"Processing video: {frames_processed}/{total_frames} frames",
                )
            status.caption(
                f"Frames processed: {frames_processed}/{total_frames}. "
                "Cumulative counts use tracker IDs and may be affected if tracking loses/reassigns an ID."
            )
    except Exception as exc:
        st.error(f"An error occurred during video processing: {exc}")
        raise
    finally:
        processing_cap.release()
        writer.release()

    if frames_processed == 0:
        st.error("No frames were processed. The video may be invalid or unsupported.")
        st.stop()

    # Keep the result in session state so it remains available after Streamlit reruns.
    with open(output_path, "rb") as processed_file:
        st.session_state["processed_video_bytes"] = processed_file.read()
    st.session_state["processed_video_name"] = f"{Path(uploaded_video.name).stem}_detected.mp4"
    st.session_state["processed_summary"] = {
        "frames": frames_processed,
        "lane_counts_sum": lane_counts_total,
        "total_unique_vehicles_passed": len(seen_vehicle_ids_any_lane),
    }
    progress.progress(1.0, text="Analysis complete")
