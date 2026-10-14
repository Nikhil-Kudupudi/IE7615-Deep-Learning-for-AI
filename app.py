from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "yolov8s_milestone2_best.pt"

st.set_page_config(page_title="YOLOv8s Object Detection", layout="wide")
st.title("YOLOv8s Object Detection")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 0.9, 0.45, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    result = model.predict(image, conf=conf, iou=iou, verbose=False)[0]

    # plot() returns a BGR numpy array; flip to RGB for display
    annotated = result.plot()[:, :, ::-1]

    left, right = st.columns(2)
    left.subheader("Original")
    left.image(image, use_container_width=True)
    right.subheader("Prediction")
    right.image(annotated, use_container_width=True)

    st.subheader(f"Detections ({len(result.boxes)})")
    if len(result.boxes):
        rows = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append(
                {
                    "class": result.names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "x1": round(x1), "y1": round(y1),
                    "x2": round(x2), "y2": round(y2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No objects detected. Try lowering the confidence threshold.")
