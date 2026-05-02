import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import os
import tempfile
from PIL import Image

# --- 1. SETTINGS & UI CONFIGURATION ---
st.set_page_config(page_title="Visionary AI Glasses", layout="wide", page_icon="🕶️")

# Custom CSS for a Premium Dark Mode Experience
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-image: linear-gradient(to right, #4A90E2, #63B3ED);
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); filter: brightness(1.1); }
    .status-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1e2130;
        border-left: 6px solid #4A90E2;
        margin-bottom: 20px;
    }
    .detection-card {
        padding: 15px;
        background: #2d3748;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid #4a5568;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 { color: #4A90E2; text-shadow: 2px 2px 4px #000000; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_vision_models():
    # Model file check
    model_path = "best_yolo_model.pt"
    if not os.path.exists(model_path):
        st.error(f"Error: {model_path} file nahi mili repo mein!")
        st.stop()
        
    yolo = YOLO(model_path)
    # EasyOCR initialization (CPU mode for Streamlit Cloud)
    reader = easyocr.Reader(['en'], gpu=False) 
    return yolo, reader

# Header section
st.markdown("<h1 style='text-align: center;'>🕶️ Smart Assistive Glasses AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0;'>Thapar University Innovation Lab Project</p>", unsafe_allow_html=True)

with st.spinner("Initializing AI Core..."):
    yolo_model, ocr_reader = load_vision_models()

# --- 3. LAYOUT DESIGN ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='status-box'><h3>📸 Visual Input</h3></div>", unsafe_allow_html=True)
    
    # Optional: camera_input bhi use kar sakte ho real-time feel ke liye
    uploaded_file = st.file_uploader("Upload an image (Scene, Text, or Currency)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file)
        # Updated: width="stretch" used instead of deprecated use_container_width
        st.image(raw_image, caption="Environment View", width="stretch")

# --- 4. CORE PROCESSING ENGINE ---
with col2:
    st.markdown("<div class='status-box'><h3>🧠 AI Analysis</h3></div>", unsafe_allow_html=True)
    
    if uploaded_file and st.button("EXECUTE SCENE ANALYSIS"):
        # Image conversion
        frame = np.array(raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        display_frame = frame.copy()
        
        with st.spinner("Processing objects and text..."):
            results = yolo_model.predict(display_frame, conf=0.25)
            boxes = results[0].boxes
        
        all_detections = []
        speech_text = ""

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_idx = int(box.cls[0])
                label = yolo_model.names[cls_idx].lower()
                conf = float(box.conf[0])
                
                # CASE A: TEXT DETECTION
                if "text" in label:
                    h, w, _ = frame.shape
                    padding = 10
                    crop = frame[max(0, y1-padding):min(h, y2+padding), max(0, x1-padding):min(w, x2+padding)]
                    
                    if crop.size > 0:
                        ocr_result = ocr_reader.readtext(crop, paragraph=True)
                        extracted_text = " ".join([res[1] for res in ocr_result])
                        
                        if extracted_text.strip():
                            all_detections.append(f"📄 **Text Identified:** {extracted_text}")
                            speech_text += f"Reading text: {extracted_text}. "
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (74, 144, 226), 4)
                
                # CASE B: CURRENCY DETECTION
                elif any(den in label for den in ["10", "20", "50", "100", "200", "500"]):
                    currency_info = f"{label} Rupee Note"
                    all_detections.append(f"💰 **Currency:** {currency_info} ({conf:.1%} confidence)")
                    speech_text += f"Detected a {currency_info}. "
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (46, 204, 113), 4)

            # Processed output display
            st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Processed Analysis", width="stretch")
            
            if all_detections:
                st.write("---")
                for item in all_detections:
                    st.markdown(f"<div class='detection-card'>{item}</div>", unsafe_allow_html=True)
                
                # Audio Feedback
                if speech_text:
                    with st.spinner("Generating Audio Feedback..."):
                        clean_audio_text = speech_text.replace("**", "")
                        tts = gTTS(text=clean_audio_text, lang='en')
                        
                        # Temp file storage for Streamlit stability
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                            tts.save(fp.name)
                            st.audio(fp.name, format="audio/mp3", autoplay=True)
            else:
                st.warning("Detection toh hui par meaningful text ya currency nahi mili.")
        else:
            st.error("AI ko scene mein kuch samajh nahi aaya. Please clear image try karein.")