# PolyVerba

**Real‑Time Multilingual Captioning & Speech Translation System**

PolyVerba is an edge-based, offline-first AI system that converts live speech or system audio into real-time text captions, and optionally translates them into various Indian languages on the fly. Designed for classrooms, seminars, accessibility support, and virtual meetings, PolyVerba acts as a universal live-captioning device that runs entirely on your local machine without relying on paid cloud APIs.

---

## What Does PolyVerba Do?

PolyVerba listens to audio playing on your computer (or spoken into your microphone) and performs the following tasks in real-time:

1. **System Audio Capture**: It strictly captures what you hear (a YouTube lecture, a Zoom meeting, or a local media file).
2. **Real-time Speech-to-Text (STT)**: It transcribes the captured English audio into text instantly.
3. **Live Machine Translation**: It instantly translates the English transcriptions into a target language of your choice (such as Hindi, Marathi, Tamil, etc.).
4. **Web-based Live Broadcaster**: It streams these captions directly to a clean, easy-to-read web interface hosted locally on your device.

By combining STT, Neural Machine Translation, and WebSockets, PolyVerba provides a robust local accessibility tool.

---

## How It Works (The Architecture)

PolyVerba’s architecture is divided into four main layers:

### 1. Audio Routing & Capture (VB-Cable)
Rather than simply listening to a microphone, the system needs to capture "System Audio". To do this without complex OS-level loopbacks, PolyVerba relies on **VB-Cable**, a virtual audio device. 
* **The Flow**: Your computer's audio is routed to the `CABLE Input` (acting as speakers). PolyVerba then listens to the `CABLE Output` (acting as a microphone). 
* The `soundcard` and `sounddevice` Python libraries continuously record this stream in memory, slicing the audio into overlapping 2-to-3 second chunks.

### 2. Speech Recognition Engine (Faster-Whisper)
Once an audio chunk is captured, it is processed by **Faster-Whisper** (a highly optimized version of OpenAI's Whisper model).
* Faster-Whisper runs the STT inference directly on your CPU (or GPU if available).
* It accurately handles background noise and accents, converting the short audio chunk into an English text segment.

### 3. Real-Time Translation (IndicTrans2)
If the user selects a target language other than English (e.g., Hindi), the transcribed English text is immediately passed to the **IndicTrans2** model (`ai4bharat/indictrans2-en-indic-dist-200M`).
* This model is specifically designed for high-accuracy translation between English and 20+ Indian languages.
* A highly-distilled 200M parameter model is used to ensure the translation happens in milliseconds, guaranteeing real-time delivery.

### 4. Web Server & Streaming (FastAPI + WebSockets)
As soon as a text chunk is finalized (transcribed and translated), it is placed into an asynchronous queue.
* A **FastAPI** web server manages the user interface.
* Background tasks read the text from the queue and instantly broadcast it via **WebSockets** to any connected browser clients.
* The frontend (HTML/JS) receives these WebSocket events and dynamically appends the newest captions to the screen, providing a seamless "live subtitling" experience in the browser.

---

## Technology Stack

| Component | Technology Used | Purpose |
| :--- | :--- | :--- |
| **Backend & API** | Python, FastAPI, Uvicorn | Manages the web server and endpoints. |
| **Real-time Comms** | WebSockets (`asyncio`) | Streams live captions from backend to the browser. |
| **Audio Capture** | `soundcard`, `sounddevice`, VB-Cable | Grabs the internal system audio. |
| **Pre-processing** | FFmpeg, `numpy`, `scipy` | Resamples and processes audio streams. |
| **Speech-to-Text** | `faster-whisper` | Offline English speech recognition. |
| **Translation** | `IndicTrans2` (via HuggingFace) | State-of-the-art En-to-Indic translation. |
| **Machine Learning**| PyTorch, Transformers | Powers the AI inference engine. |

---

## Project Structure

```text
d:\polyverba\
├── stt/
│   ├── system_audio.py      # Core logic loop: Audio recording & processing
│   ├── transcribe.py        # Faster-Whisper integration
│   └── translate.py         # IndicTrans2 inference wrapper
├── web/
│   ├── static/              # CSS / JS assets for the frontend
│   └── templates/           # HTML templates for FastAPI
├── translation/
│   └── indictrans2.py       # Isolated testing scripts for translation
├── web_server.py            # FastAPI Entry Point and WebSocket broadcaster
├── setup.md                 # Complete Installation Guide
├── indic_setup.md           # Documentation on ML env decisions
└── README.md                # This file
```

---

## Quick Start & Installation

Because PolyVerba relies on heavy ML models, **Python 3.10** is required to successfully install the dependencies (specifically `transformers` and `tokenizers`). 

For a complete, step-by-step setup guide covering virtual environments, VB-Cable configuration, and HuggingFace API setup, please see our dedicated setup document:

👉 **[Read the Complete Setup Guide (`setup.md`)](setup.md)**

### Basic Running Instructions
Once your environment is set up and activated:

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Start the web server
python web_server.py
```
Open **`http://localhost:8080`** in your browser to view the interface, select your target languages, and hit Start!

---

## Future Roadmap

- Enhancing the dynamic styling and typography of the live captions web UI.
- Support for Cloud Fallback (using APIs if the local model fails).
- Direct speech-to-speech output using Coqui TTS for translated audio playback.
- Creating a shareable QR code for multiple users on the same local network to view captions on their mobile phones.

## License
Created for academic, research, and accessibility purposes.
