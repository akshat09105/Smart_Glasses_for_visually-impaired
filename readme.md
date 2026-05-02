# 🕶️ SmartVision AI: Assistive Vision Glasses

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://ultralytics.com/)
[![Live Demo](https://img.shields.io/badge/Live-App-green?style=flat&logo=streamlit)](https://smartglassesforvisually-impaired-dbmnsfmdg7wkagqlucvawt.streamlit.app/)

An AI-powered assistive vision system designed for the visually impaired. This project implements a cascaded deep learning pipeline that localizes objects of interest (Text and Currency) and provides real-time auditory feedback.

---

## 🚀 The Core Pipeline

The system operates on a first-principles approach to scene understanding:

1.  **Localization (YOLOv8):** Scans the environment to detect `text_block` and `currency` denominations.
2.  **Recognition (EasyOCR):** The detected regions are cropped and pre-processed. To handle small or pixelated text (like on billboards), we apply an **Upscaling factor of 1.5x** with **Cubic Interpolation**.
3.  **Auditory Feedback (gTTS):** Converts the processed text into natural-sounding speech, providing an intuitive interface for the user.

---

## 🔗 Live Access
You can interact with the live application here:
👉 [**Launch SmartVision AI App**](https://smartglassesforvisually-impaired-dbmnsfmdg7wkagqlucvawt.streamlit.app/)

---

## ✨ Features

* **Intelligent Reading Order:** Implements $y$-axis sorting of bounding boxes to ensure text is read from top-to-bottom.
* **Cascaded OCR Logic:** Uses a localized cropping mechanism to minimize background noise and maximize recognition accuracy.
* **Currency Detection:** Specifically trained to identify Indian Currency denominations.
* **Accessibility UI:** High-contrast Streamlit interface designed for ease of use.
* **Voice Auto-play:** Seamless transition from image analysis to speech output.

---

## 🛠️ Technical Implementation

### **Cropping & Pre-processing Logic**
To prevent character cutoff during OCR, we apply a padding $p$ to the YOLO bounding box $(x_1, y_1, x_2, y_2)$:

$$Crop = Image[y_1-p : y_2+p, \space x_1-p : x_2+p]$$

This is followed by an interpolation step to enhance character edges:
```python
crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
