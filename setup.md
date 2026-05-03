# PolyVerba Complete Setup Guide

This guide details exactly how to set up the PolyVerba project from scratch on a new machine. It covers everything from system-level audio routing to Python ML dependencies and HuggingFace authentication.

## Prerequisites
Before you start, ensure you have the following installed on your machine (Windows 10/11 is assumed):
- **Python 3.10**: This specific version is **highly recommended** to avoid compilation issues with Rust and `tokenizers`.
- **Git for Windows**: Required for HuggingFace to store its authentication tokens properly on Windows.
- **VB-Cable**: Used to route system audio (speakers) into a virtual microphone.
- **FFmpeg**: Required by Whisper for audio processing.

---

## Step 1: Install & Configure VB-Cable (Audio Routing)
PolyVerba captures your computer's system audio (e.g., a YouTube video or Zoom meeting) by routing it through a virtual audio cable.

1. Download and install VB-Cable from: [https://vb-audio.com/Cable/](https://vb-audio.com/Cable/)
2. Run `VBCABLE_Setup_x64.exe` as Administrator and restart your PC.
3. Open Windows Sound Settings (`Windows Key + R` -> type `mmsys.cpl` -> hit Enter).
4. **Playback Tab**: Right-click `CABLE Input` and select **Set as Default Device**.
5. **Recording Tab**: Right-click `CABLE Output` and select **Set as Default Device**.
6. **Hear Audio**: While on the Recording Tab, right-click `CABLE Output` -> **Properties** -> **Listen** tab. Check **"Listen to this device"** and select your physical Speakers/Headphones from the dropdown. This allows you to still hear your laptop's audio.

---

## Step 2: Install FFmpeg
1. Download a pre-compiled Windows build of FFmpeg from: [https://www.gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/)
2. Extract the archive.
3. Add the extracted `bin` folder (containing `ffmpeg.exe`) to your system's `PATH` environment variable.
4. Open a new terminal and verify with: `ffmpeg -version`

---

## Step 3: Project Setup (Virtual Environment)
Open your terminal (Command Prompt/PowerShell) and navigate to your project directory.

1. **Create the Virtual Environment**:
   It is crucial to use Python 3.10 to prevent ML library compilation errors.
   ```bash
   py -3.10 -m venv venv
   ```

2. **Activate the Environment**:
   ```bash
   venv\Scripts\activate
   ```

---

## Step 4: Install Dependencies
With the virtual environment activated, install all required Python packages. Note that specific versions of Transformers are needed to maintain compatibility with IndicTrans2.

```bash
pip install --upgrade pip

# Core ML and Translation dependencies
pip install torch
pip install transformers==4.37.2
pip install sentencepiece sacremoses indic-nlp-library accelerate deep-translator

# Audio capturing and Whisper dependencies
pip install faster-whisper soundcard sounddevice numpy scipy ffmpeg-python

# Web Server dependencies
pip install fastapi uvicorn jinja2 websockets
```

---

## Step 5: HuggingFace Authentication (For IndicTrans2)
The translation model `ai4bharat/indictrans2-en-indic-dist-200M` is hosted on HuggingFace as a gated repository, which means you must authenticate before downloading it.

1. Go to [HuggingFace](https://huggingface.co/) and create an account.
2. Visit the model page: `ai4bharat/indictrans2-en-indic-dist-200M` and **agree to the usage terms**.
3. Go to your HuggingFace Settings -> Access Tokens, and generate a new **Read** token.
4. Open your terminal (ensure Git is installed) and run:
   ```bash
   huggingface-cli login
   ```
5. Paste your token (it will be invisible) and hit Enter. *Note: If this throws `WinError 2`, you are missing Git on your system path.*

---

## Step 6: Running the Application

You have two main ways to run PolyVerba:

### Option A: Web Interface (Recommended)
This launches a FastAPI server with a web UI to view captions.
1. Make sure your virtual environment is activated.
2. Run the server:
   ```bash
   python web_server.py
   ```
3. Open your browser and navigate to `http://localhost:8080`.
4. Play any system audio (e.g., a video). Select your source/target languages and model, then hit start.

### Option B: Terminal Mode
If you only want live captions printed directly to the terminal:
1. Make sure your virtual environment is activated.
2. Run:
   ```bash
   run_captions.bat
   # or natively: python -m stt.system_audio
   ```
