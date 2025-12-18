import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import time


# -------------------------------
# 🎨 UI Enhancement: Custom CSS & Dark Anime Background
# -------------------------------
def apply_custom_styles():
    BG_URL = "https://i.etsystatic.com/54143431/r/il/48d57c/6256676432/il_fullxfull.6256676432_r6vw.jpg"

    st.markdown(f"""
    <style>
    /* Dark Theme & Anime Background */
    .stApp {{
        background-image: url("{BG_URL}");
        background-attachment: fixed;
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        background-color: rgba(0, 0, 0, 0.82);
    }}

    /* Main Content Wrapper - Dark Setting */
    .main-box {{
        background: rgba(15, 23, 42, 0.88);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;

        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.6);
        margin-bottom: 10px;
        width: 100%;
    }}

    /* Light Text Legibility */
    h1, h2, h3, h4, p, span, label, div[data-testid="stMarkdownContainer"] > p {{
        color: #f1f5f9 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.9);
    }}

    /* Emotion Result Card - Forces Below Image */
    .emotion-card {{
        background: rgba(30, 41, 59, 0.95);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        border-top: 6px solid #3b82f6;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.5);
        margin-top: 20px;
        width: 100%;
    }}

    /* UPDATED: Sidebar - Blurry and Transparent */
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.07) !important;
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}

    /* Ensure sidebar text remains readable */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
        color: #ffffff !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="FER AI | Emotion Detector",
    page_icon="🎭",
    layout="centered"
)

apply_custom_styles()

# Helper: Mapping Emotions to Emojis
EMOJI_MAP = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😊", "sad": "😢", "surprise": "😲", "neutral": "😐"
}

# -------------------------------
# Header Section
# -------------------------------
st.title("🎭 FER AI")
st.markdown("#### Advanced Facial Emotion Recognition System")

# -------------------------------
# Upload Section
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload a face image to decode emotions", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Image Processing
    image = Image.open(uploaded_file).convert("RGB")

    # 2. Display Uploaded Image (Full Width)
    st.image(image, caption="Uploaded Subject", use_column_width=True)

    # Temporary storage for DeepFace
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        img_path = tmp.name

    with st.spinner("🧠 Analyzing facial micro-expressions..."):
        try:
            start_time = time.time()
            # DeepFace Analysis
            result = DeepFace.analyze(img_path, actions=["emotion"], enforce_detection=False)
            inference_time = time.time() - start_time

            emotions = result[0]["emotion"]
            dominant = result[0]["dominant_emotion"]
            emoji = EMOJI_MAP.get(dominant, "😶")

            # 3. Display Result Card BELOW the image
            st.markdown(f"""
            <div class="emotion-card">
                <h1 style="font-size: 100px; margin:0;">{emoji}</h1>
                <h1 style="margin:0; text-transform: uppercase; letter-spacing: 3px; color: #3b82f6 !important;">{dominant}</h1>
                <p style="color: #94a3b8 !important; font-size: 1.2em;">Confidence: {emotions[dominant]:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            # 4. Probability Distribution (Integrated directly to avoid empty box gaps)
            st.markdown('<div class="main-box">', unsafe_allow_html=True)
            st.markdown("### 📊 Emotion Probability Distribution")
            sorted_ems = dict(sorted(emotions.items(), key=lambda item: item[1], reverse=True))

            for emo, score in sorted_ems.items():
                st.write(f"**{emo.capitalize()}**")
                st.progress(float(score) / 100)

            st.caption(f"Analysis Time: {inference_time:.3f}s")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error("⚠️ Face detection failed. Ensure the face is clear.")

# -------------------------------
# Technical Info Section (Removed <br> to prevent empty space)
# -------------------------------
with st.expander("🔍 How the FER Model Works"):
    st.markdown("""
    * **DeepFace Engine:** Uses the VGG-Face or ResNet architecture to extract facial landmarks.
    * **Emotion Head:** A pre-trained dense network that classifies the geometry of lips, eyes, and eyebrows into 7 key emotions.
    * **Micro-expressions:** The model captures subtle variations in facial muscles to calculate a probability score for each emotion.
    """)

with st.sidebar:
    st.header("📌 Project Details")
    st.info(
        "**Backend:** Keras & TensorFlow\n\n**Frontend:** Streamlit Glassmorphism\n\n**Task:** Multiclass Emotion Classification")
    st.warning("⚠️ Ensure the subject is facing the camera directly for best accuracy.")
    st.warning("⚠️ Results depend on image clarity and lighting.")
