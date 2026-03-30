import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import base64

# 🔥 Function to set background
def set_bg(image_file):
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

        /* Make all text white for dark theme */
        h1, h2, h3, h4, p, div {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ✅ Apply background (this calls the function)
set_bg("Media/img1.jpg")

# Load model
model = YOLO("best.pt")

# Title
st.title("🚧 TAS Pothole Detection System")

# Team info
st.markdown("""
### 👨‍💻 Developed By:
- **Tanveer Singh Bindra** (21CSU124)  
- **Suyash Rai** (21CSU464)  
- **Ankesh** (22CSU020)  

### 🎓 Supervised By:
- **Dr. Shilpa Mahajan**
""")



st.write("---")

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Original Image")

    img = np.array(image)

    results = model(img)
    result_img = results[0].plot()

    st.image(result_img, caption="🚧 Detected Potholes")

    # 🔗 Load GitHub image
def get_base64_image(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

github_img = get_base64_image("Media/github.png")

# 🔥 Centered clickable GitHub button
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