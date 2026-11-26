from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import cv2
import numpy as np
import streamlit as st

from detectit.config import load_config
from detectit.models.yolo import Detector
from detectit.utils.drawing import class_counts, draw_detections

IMAGE_TYPES = ("jpg", "jpeg", "png", "webp")
VIDEO_TYPES = ("mp4", "mov", "avi", "mkv", "webm")


@st.cache_resource(show_spinner=False)
def load_detector(config, checkpoint_version):
    return Detector(config), Lock()


@st.cache_resource(max_entries=8, show_spinner=False)
def store_video(digest, suffix, _data):
    temporary = TemporaryDirectory(prefix="detectit-")
    path = Path(temporary.name) / f"video{suffix}"
    path.write_bytes(_data)
    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return temporary, path, count


def read_frame(path, frame):
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        return image if ok else None
    finally:
        capture.release()


def main():
    st.set_page_config(page_title="DetectIt", layout="wide")
    st.title("DetectIt")
    with st.sidebar:
        options = {
            path.stem: path for path in sorted(Path("configs").glob("*.yaml"))
            if "weights" in load_config(path) and "training" not in load_config(path)
        }
        selected = st.selectbox("Checkpoint", [None, *options], index=(
            list(options).index("yolo11n-coco") + 1 if "yolo11n-coco" in options else 0
        ), format_func=lambda value: value or "No checkpoint")
        config = load_config(options[selected]) if selected else None
        if config:
            confidence = st.slider("Confidence", 0.0, 1.0, float(config["confidence"]), 0.01, key=f"{selected}-confidence")
            iou = st.slider("NMS IoU", 0.0, 1.0, float(config["nms_iou"]), 0.01, key=f"{selected}-iou")
            st.caption(f"{config['model']} · {config['dataset']['name']} · {options[selected].name}")
    mode = st.radio("Source", ("Upload", "Camera"), horizontal=True)
    upload = (
        st.file_uploader("Choose an image or video", type=(*IMAGE_TYPES, *VIDEO_TYPES))
        if mode == "Upload" else st.camera_input("Take a photo")
    )
    if upload is None:
        return
    data = upload.getvalue()
    suffix = Path(upload.name).suffix.lower()
    if suffix.removeprefix(".") in VIDEO_TYPES:
        digest = sha256(data).hexdigest()
        temporary, path, count = store_video(digest, suffix, data)
        frame = st.slider("Frame", 1, count, 1, key=f"{digest}-frame") - 1 if count > 1 else 0
        image = read_frame(path, frame)
    else:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR) if data else None
    if image is None:
        st.error("OpenCV could not decode this image or video frame.")
        return
    if config is None:
        st.info("Choose a checkpoint to run detection. Your source is shown below.")
        st.image(image, channels="BGR", width="stretch")
        return

    weights = Path(config["weights"])
    if weights.is_file():
        stat = weights.stat()
        checkpoint_version = (stat.st_mtime_ns, stat.st_size)
    else:
        checkpoint_version = None
    try:
        with st.spinner("Detecting objects…"):
            detector, lock = load_detector(config, checkpoint_version)
            with lock:
                detections = detector.predict(image, confidence=confidence, nms_iou=iou)
    except (OSError, RuntimeError, ImportError) as error:
        st.warning(f"Predictions need a working checkpoint and the DetectIt dependencies. {error}")
        st.image(image, channels="BGR", width="stretch")
        return

    preview, summary = st.columns((3, 1))
    preview.image(draw_detections(image, detections), channels="BGR", width="stretch")
    with summary:
        st.subheader("Objects in this frame")
        counts = class_counts(detections)
        if counts:
            st.table([{"Class": label, "Count": count} for label, count in counts.items()])
        else:
            st.info("No objects matched these thresholds.")


if __name__ == "__main__":
    main()
