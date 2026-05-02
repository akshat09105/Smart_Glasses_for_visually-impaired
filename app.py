import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
from gtts import gTTS
import os
from PIL import Image

# --- 1. SETTINGS & BEAUTIFUL UI ---
st.set_page_config(page_title="Visionary AI Glasses", layout="wide", page_icon="🕶️")

# Custom CSS for Hackathon Winning Frontend
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
    h1 { color: #4A90E2; font-family: 'Inter', sans-serif; text-shadow: 2px 2px 4px #000000; }
    .detection-card {
        padding: 15px;
        background: #2d3748;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid #4a5568;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🕶️ Smart Assistive Glasses AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0;'>Thapar University Innovation Lab Project</p>", unsafe_allow_html=True)

# --- 2. MODEL LOADING (CACHED) ---
@st.cache_resource
def load_vision_models():
    # Sahi file name yahan load ho rahi hai
    yolo = YOLO("best_yolo_model.pt")
    # EasyOCR model (Assuming models folder exists or it will download)
    reader = easyocr.Reader(['en'], model_storage_directory='./models')
    return yolo, reader

with st.spinner("Wait karo bhai, AI dimaag load ho raha hai..."):
    yolo_model, ocr_reader = load_vision_models()

# --- 3. LAYOUT DESIGN ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='status-box'><h3>📸 Input Frame</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Image upload karo (Book page, signage, ya notes)", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        raw_image = Image.open(uploaded_file)
        st.image(raw_image, caption="Current View", use_container_width=True)

# --- 4. CORE LOGIC: DETECTION + OCR + SPEECH ---
with col2:
    st.markdown("<div class='status-box'><h3>🧠 Smart Insights</h3></div>", unsafe_allow_html=True)
    
    if uploaded_file and st.button("RUN SCENE ANALYSIS"):
        # Image Processing for OpenCV
        frame = np.array(raw_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # YOLO Detection
        with st.spinner("Objects dhoond raha hoon..."):
            results = yolo_model.predict(frame, conf=0.20) # Lower conf for better text recall
            boxes = results[0].boxes
        
        all_detected_text = []
        speech_queue = ""

        if len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_idx = int(box.cls[0])
                label = yolo_model.names[cls_idx].lower() # Handle "text" vs "text_block"
                
                # --- LOGIC 1: TEXT PROCESSING ---
                if "text" in label:
                    # Padding add karo taaki EasyOCR border characters na miss kare
                    h, w, _ = frame.shape
                    padding = 10
                    crop = frame[max(0, y1-padding):min(h, y2+padding), max(0, x1-padding):min(w, x2+padding)]
                    
                    if crop.size > 0:
                        with st.spinner("OCR Reading..."):
                            # EasyOCR on crop
                            ocr_result = ocr_reader.readtext(crop, paragraph=True) # Paragraph mode is better for flow
                            text_content = " ".join([res[1] for res in ocr_result])
                            
                            if text_content.strip():
                                all_detected_text.append(f"📄 **Text Block:** {text_content}")
                                speech_queue += text_content + ". "
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (74, 144, 226), 4)
                
                # --- LOGIC 2: CURRENCY PROCESSING ---
                elif any(den in label for den in ["10", "20", "50", "100", "200", "500"]):
                    currency_msg = f"{label} Rupee note"
                    all_detected_text.append(f"💰 **Currency:** {currency_msg}")
                    speech_queue += currency_msg + ". "
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (46, 204, 113), 4)

            # Display Result Image
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed View", use_container_width=True)
            
            # Show Text Findings & Play Audio
            if speech_queue:
                st.write("---")
                for text_msg in all_detected_text:
                    st.markdown(f"<div class='detection-card'>{text_msg}</div>", unsafe_allow_html=True)
                
                # gTTS Speech Generation
                with st.spinner("Audio is getting prepared be patient!"):
                    # Cleaning text for gTTS (Removing special markdown)
                    clean_text = speech_queue.replace("**", "").replace("_", "")
                    tts = gTTS(text=clean_text, lang='en')
                    tts.save("vision_audio.mp3")
                    st.audio("vision_audio.mp3", format="audio/mp3", autoplay=True)
            else:
                st.warning("No text inside.")
        else:
            st.error("text is not cleared to model. trying something else")

# --- FOOTER ---
st.markdown("<br><hr><p style='text-align: center; color: #718096;'>Thapar Visionary Assistive Technology v1.0</p>", unsafe_allow_html=True)