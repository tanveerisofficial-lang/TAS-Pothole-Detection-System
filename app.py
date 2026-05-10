import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import keras
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import os


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="TAS Pothole Detection System | Hybrid Model",
    page_icon="🚧",
    layout="wide"
)


# =========================
# BACKGROUND FUNCTION
# =========================
def set_bg(image_file):
    if not os.path.exists(image_file):
        return

    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
            linear-gradient(
                to bottom,
                rgba(0, 0, 0, 0) 0%,
                rgba(0, 0, 0, 0.2) 50%,
                rgba(0, 0, 0, 0.5) 75%,
                rgba(5, 10, 25, 0.9) 90%,
                rgba(5, 10, 25, 1) 100%
            ),
            url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        h1, h2, h3, h4, p, div, label {{
            color: white !important;
        }}

        .stSlider label {{
            color: white !important;
        }}

        .stFileUploader label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_bg("Media/img1.jpg")


# =========================
# MODEL PATHS
# =========================
YOLO_MODEL_PATH = "best.pt"
CNN_MODEL_PATH = "latest_hybrid_model.h5"


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)


@st.cache_resource
def load_cnn_model():
    return keras.models.load_model(
        CNN_MODEL_PATH,
        compile=False,
        safe_mode=False
    )


try:
    yolo_model = load_yolo_model()
    cnn_model = load_cnn_model()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Model loading failed: {e}")


# =========================
# HYBRID PREDICTION FUNCTION
# =========================
def hybrid_predict(
    image_pil,
    yolo_conf_threshold=0.25,
    cnn_refinement_threshold=0.5
):
    """
    Stage 1: YOLO detects candidate pothole boxes.
    Stage 2: CNN checks each cropped YOLO box.
    Stage 3: Only CNN-confirmed boxes are shown.
    """

    image_rgb = image_pil.convert("RGB")
    image_np = np.array(image_rgb)

    # Stage 1: YOLO Scout
    results = yolo_model.predict(
        image_np,
        conf=yolo_conf_threshold,
        verbose=False
    )

    boxes = results[0].boxes.xyxy.cpu().numpy()

    confirmed_detections = []

    # Stage 2: CNN Judge
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)

        # Keep coordinates inside image bounds
        h, w, _ = image_np.shape
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_np[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # CNN was trained on 224x224 images
        crop_pil = Image.fromarray(crop).resize((224, 224))
        crop_array = np.expand_dims(np.array(crop_pil), axis=0)

        cnn_confidence = float(cnn_model.predict(crop_array, verbose=0)[0][0])

        # Stage 3: Final Verdict
        if cnn_confidence >= cnn_refinement_threshold:
            confirmed_detections.append({
                "box": [x1, y1, x2, y2],
                "cnn_confidence": cnn_confidence
            })

    return confirmed_detections, boxes


# =========================
# DRAW FINAL DETECTIONS
# =========================
def draw_detections(image_pil, detections):
    output_img = image_pil.convert("RGB").copy()
    draw = ImageDraw.Draw(output_img)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        confidence = det["cnn_confidence"]

        # Draw final confirmed box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=(217, 255, 0),
            width=4
        )

        label = f"Pothole: {confidence:.2f}"

        # Label background
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        label_y = max(0, y1 - text_h - 8)

        draw.rectangle(
            [(x1, label_y), (x1 + text_w + 10, label_y + text_h + 8)],
            fill=(0, 0, 0)
        )

        draw.text(
            (x1 + 5, label_y + 4),
            label,
            fill=(217, 255, 0),
            font=font
        )

    return output_img


# =========================
# APP UI
# =========================
st.title("🚧 TAS Pothole Detection System | Hybrid Model")

st.markdown("""
### 👨‍💻 Developed By:
- **Tanveer Singh Bindra** (21CSU124)  
- **Suyash Rai** (21CSU464)  
- **Ankesh** (22CSU020)  

### 🎓 Supervised By:
- **Dr. Shilpa Mahajan**
""")

st.write("---")

st.markdown("""
## Hybrid Detection Pipeline

This system uses a **YOLO + CNN hybrid model**:

**Stage 1 — YOLO Scout:**  
YOLOv8-Nano quickly scans the full image and finds possible pothole regions.

**Stage 2 — CNN Judge:**  
Each YOLO box is cropped and passed to the custom CNN classifier.

**Stage 3 — Final Verdict:**  
Only potholes confirmed by the CNN are shown in the final output.
""")

st.write("---")


# =========================
# MODEL STATUS
# =========================
if models_loaded:
    st.success("✅ Hybrid model loaded successfully: YOLO Scout + CNN Judge")
else:
    st.stop()


# =========================
# THRESHOLD CONTROLS
# =========================
with st.expander("⚙️ Advanced Detection Settings"):
    yolo_conf = st.slider(
        "YOLO Candidate Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

    cnn_conf = st.slider(
        "CNN Refinement Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.50,
        step=0.05
    )


# =========================
# IMAGE UPLOAD
# =========================
uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🖼️ Original Image", use_container_width=True)

    with st.spinner("Running hybrid detection..."):
        confirmed_detections, yolo_boxes = hybrid_predict(
            image,
            yolo_conf_threshold=yolo_conf,
            cnn_refinement_threshold=cnn_conf
        )

        final_img = draw_detections(image, confirmed_detections)

    with col2:
        st.image(
            final_img,
            caption="🚧 Hybrid Confirmed Potholes",
            use_container_width=True
        )

    st.write("---")

    st.subheader("📊 Detection Summary")

    st.write(f"**YOLO Candidate Detections:** {len(yolo_boxes)}")
    st.write(f"**CNN Confirmed Potholes:** {len(confirmed_detections)}")

    if len(yolo_boxes) > 0:
        refinement_rate = (len(confirmed_detections) / len(yolo_boxes)) * 100
        st.write(f"**Final Acceptance Rate:** {refinement_rate:.2f}%")

    if len(confirmed_detections) == 0:
        st.warning("No potholes were confirmed by the hybrid model.")
    else:
        st.success(f"{len(confirmed_detections)} pothole(s) confirmed by the hybrid system.")

    with st.expander("🔍 Confirmed Detection Details"):
        for i, det in enumerate(confirmed_detections, start=1):
            st.write(
                f"**Detection {i}:** "
                f"Box = {det['box']}, "
                f"CNN Confidence = {det['cnn_confidence']:.4f}"
            )


# =========================
# GITHUB BUTTON
# =========================
def get_base64_image(path):
    if not os.path.exists(path):
        return None

    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()


github_img = get_base64_image("Media/github.png")

if github_img:
    st.markdown(
        f"""
        <div style="text-align:center; margin-top:20px;">
            <a href="https://github.com/tanveerisofficial-lang/TAS-Pothole-Detection-System" target="_blank">
                <div style="
                    display:inline-flex;
                    align-items:center;
                    gap:10px;
                    padding:10px 18px;
                    background-color:white;
                    border-radius:12px;
                    box-shadow:0 4px 12px rgba(0,0,0,0.3);
                    transition:0.3s;
                "
                onmouseover="this.style.transform='scale(1.05)'"
                onmouseout="this.style.transform='scale(1)'"
                >
                    <img src="data:image/png;base64,{github_img}" width="32">
                    <span style="color:black; font-weight:bold; font-size:15px;">
                        View on GitHub
                    </span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
