# IndicTrans2 Setup – What We Did and Why We Did It

This document explains the complete setup process of IndicTrans2 for the PolyVerba project, including the reasoning behind each step.

The goal was to enable real-time translation of Whisper-generated English captions into Indian languages (starting with Hindi).

---

# 1. Objective

We wanted to extend the PolyVerba pipeline:

System Audio → Whisper → English Text

into:

System Audio → Whisper → English Text → IndicTrans2 → Hindi Captions

This required installing and configuring IndicTrans2 locally.

---

# 2. Why IndicTrans2?

We chose IndicTrans2 because:

- It is built specifically for Indian languages
- It supports 20+ Indian languages
- It works offline (no API cost)
- It provides better Hindi-English translation quality compared to generic models
- It is suitable for academic and research projects

For real-time CPU usage, we selected:

ai4bharat/indictrans2-en-indic-dist-200M

Reason:
- Smaller model size
- Faster inference
- Lower RAM usage
- Suitable for laptop demo environments

---

# 3. HuggingFace Authentication (Why Required)

IndicTrans2 is hosted on HuggingFace as a gated repository.

This means:
- We must accept usage terms
- We must authenticate using a HuggingFace token

What we did:
- Created a HuggingFace account
- Generated a Read token
- Logged in using Python:

```python
from huggingface_hub import login
login()
Why:
Without authentication, model download fails with a "gated repository" error.

4. Windows Git Requirement (Why It Was Needed)
During login, HuggingFace attempts to store the token using Git credentials.

We installed Git because:

HuggingFace uses git config

Without Git, login throws WinError 2

After installing Git and adding it to PATH, login worked successfully.

5. Transformers Version Compatibility Issue
Error encountered:

AttributeError: IndicTransTokenizer has no attribute _special_tokens_map

Cause:

Latest transformers version was incompatible with IndicTrans2

Solution:
We downgraded transformers to a compatible version:

pip uninstall transformers tokenizers -y
pip install transformers==4.37.2
Why:
IndicTrans2 was built using an older Transformers API.
Newer versions changed tokenizer internals.

6. Python Version Problem (Very Important)
We initially used Python 3.13.

Problem:

tokenizers package had no prebuilt wheel for Python 3.13

pip tried to compile from source

Compilation required Rust (which caused failure)

Solution:
We installed Python 3.10 and created a new virtual environment:

py -3.10 -m venv venv
Why:

ML libraries are stable on Python 3.9–3.11

Precompiled wheels are available

Avoids Rust compilation errors

This was the most critical fix.

7. Final Working Environment
After recreating the environment with Python 3.10, we installed:

pip install torch
pip install transformers==4.37.2
pip install sentencepiece
pip install sacremoses
pip install indic-nlp-library
pip install accelerate
pip install deep-translator
pip install soundcard sounddevice numpy scipy ffmpeg-python
Then ran:

python translation/indictrans2.py
The model downloaded successfully and produced Hindi output.

8. Final Working Pipeline
The working PolyVerba translation system now functions as:

System Audio
↓
Whisper Speech Recognition
↓
English Text
↓
IndicTrans2 Translation
↓
Hindi Captions

9. Key Learnings
Use stable Python versions (3.10 recommended for ML).

Always match Transformers version with model compatibility.

HuggingFace gated models require authentication.

Windows often requires Git installation for credential storage.

Avoid bleeding-edge Python versions for production ML work.

10. Why This Setup Matters
This setup transforms PolyVerba from:

"Speech-to-Text Tool"

into:

"Real-Time Multilingual Captioning System"

It enables:

Bilingual live captions

Classroom translation

Multilingual event accessibility

Foundation for speech-to-speech translation