import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import os
from PIL import Image

# ==========================================
# 1. SETTINGS & STYLING
# ==========================================
st.set_page_config(page_title="Visionary AI Glasses", layout="wide", page_icon="🕶️")

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
    .stButton>button:hover { transform: scale(1.02); filter: brightness(1.1); border: none; }
    .status-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #1e2130;
        border-left: 6px solid #4A90E2;
        margin-bottom: 25px;
    }
    .detection-card {
        padding: 15px;
        background: #2d3748;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid #4a5568;
        color: #e2e8f0;
    }
    h1 { color: #4A90E2; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🕶️ Smart Assistive Glasses AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0;'>Assistive Technology Project | Thapar Institute</p>", unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING
# ==========================================
@st.cache_resource
def load_vision_models():
    yolo = YOLO("best_yolo_model.pt")
    reader = easyocr.Reader(['en'], model_storage_directory='./models')
    return yolo, reader

with st.spinner("AI Models load ho rahe hain..."):
    yolo_model, ocr_reader = load_vision_models()

# ==========================================
# 3. UI LAYOUT
# ==========================================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='status-box'><h3>📸 Camera Input</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Image upload karo...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, caption="Uploaded Image", use_container_width=True)

# ==========================================
# 4. CORE LOGIC
# ==========================================
with col2:
    st.markdown("<div class='status-box'><h3>🧠 AI Scene Analysis</h3></div>", unsafe_allow_html=True)
    
    if uploaded_file and st.button("START SCANNING"):
        frame = np.array(raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        with st.spinner("Analyzing scene..."):
            results = yolo_model.predict(frame, conf=0.20)
            boxes_data = results[0].boxes
        
        if len(boxes_data) > 0:
            # Sort boxes top-to-bottom
            sorted_indices = np.argsort([b.xyxy[0][1].item() for b in boxes_data])
            
            all_detected_text = []
            speech_queue = ""

            for idx in sorted_indices:
                box = boxes_data[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_idx = int(box.cls[0])
                label = yolo_model.names[cls_idx].lower()
                
                # Logic for Text
                if "text" in label:
                    h, w, _ = frame.shape
                    p = 15 # Padding
                    crop = frame[max(0, y1-p):min(h, y2+p), max(0, x1-p):min(w, x2+p)]
                    
                    if crop.size > 0:
                        # Upscale for better OCR
                        crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                        ocr_res = ocr_reader.readtext(crop, paragraph=True)
                        text_found = " ".join([res[1] for res in ocr_res])
                        
                        if text_found.strip():
                            all_detected_text.append(f"📄 **Text Block:** {text_found}")
                            speech_queue += text_found + ". " # FIXED: match the variable name
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (74, 144, 226), 4)

                # Logic for Currency
                elif any(den in label for den in ["10", "20", "50", "100", "200", "500"]):
                    currency_msg = f"{label} Rupee note"
                    all_detected_text.append(f"💰 **Currency:** {currency_msg}")
                    speech_queue += currency_msg + ". " # FIXED: match the variable name
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 4)

            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            if speech_queue:
                st.write("---")
                for item in all_detected_text:
                    st.markdown(f"<div class='detection-card'>{item}</div>", unsafe_allow_html=True)
                
                # Audio Generation
                with st.spinner("Speaking..."):
                    clean_text = speech_queue.replace("**", "").strip()
                    tts = gTTS(text=clean_text, lang='en')
                    tts.save("vision_voice.mp3")
                    st.audio("vision_voice.mp3", format="audio/mp3", autoplay=True)
            else:
                st.warning("Detection toh hui par content nahi mil raha.")
        else:
            st.error("Kuch bhi detect nahi hua.")

st.markdown("<br><hr><p style='text-align: center; color: #718096;'>Project by Akshat Gupta | TIET</p>", unsafe_allow_html=True)