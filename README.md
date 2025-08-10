# Crop Advisory — AI-Powered Plant Disease & Weather Assistant (Urdu)

## Overview
Crop Advisory is an AI-powered agricultural assistant for farmers, agronomists, and agricultural extension workers.  
It combines plant disease detection from leaf images with localized weather insights in Urdu to support decisions on crop health, irrigation, and pest prevention.

**It provides:**
- Real-time plant disease detection from leaf images
- Current weather conditions in Urdu
- Tomorrow’s weather forecast in Urdu
- (Optional in your codebase) Urdu voice input with Whisper and Urdu TTS responses

---

## Key Features
- **Plant Disease Detection** — Uses Roboflow’s hosted model to identify diseases from leaf images with confidence scores.
- **Current Weather (Urdu)** — Temperature, humidity, and conditions in Urdu via OpenWeatherMap.
- **Tomorrow’s Forecast (Urdu)** — Midday forecast details for the next day.
- **Simple Web UI** — Gradio-based interface that runs in a browser.
- **Extensible** — Hooks for RAG/embeddings and Urdu voice features are present in code if you enable them.

---

## Tech Stack

| Component            | Technology |
|---------------------|-----------|
| Programming Language| Python |
| UI                  | Gradio |
| Disease Detection   | Roboflow Inference API (`inference-sdk`) |
| Weather Data        | OpenWeatherMap API (`requests`) |
| LLM (optional prompts) | Groq API (Llama 3 family) |
| OCR/RAG (optional)  | Tesseract OCR, LangChain, FAISS, HuggingFace Embeddings |
| Audio (optional)    | OpenAI Whisper (ASR), gTTS (Urdu TTS) |



---


