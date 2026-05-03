# IndicTrans2 Setup Guide (PolyVerba)

This document explains how to install and use **IndicTrans2** locally and connect it with the PolyVerba Whisper caption system.

After completing this guide your pipeline will be:

System Audio → Whisper → English Text → IndicTrans2 → Hindi (or other Indian language) captions

No internet or paid APIs are required after the first download.

---

## 1. Prerequisites

You must already have:

- PolyVerba project folder
- Virtual environment created
- Whisper system audio transcription working

Your folder should look like:

polyverba/
│
├── stt/
├── translation/
├── venv/
├── README.md


---

## 2. Activate Virtual Environment

Open terminal inside the project folder.

cd polyverba
venv\Scripts\activate


You should see:

(venv) C:\polyverba>


---

## 3. Install Required Libraries

Run the following commands one by one:

pip install --upgrade pip
pip install torch
pip install transformers
pip install sentencepiece
pip install sacremoses
pip install indic-nlp-library
pip install accelerate
pip install gitpython


This installs the HuggingFace translation framework.

---

## 4. Prepare Indic NLP Resources

Create a folder:

mkdir indic_resources


Then run:

python -c "from indicnlp import common; common.set_resources_path('indic_resources')"


No output means success.

---

## 5. Create Translation Module

Create a folder if not present:

polyverba/translation


Create file:

translation/indictrans2.py


---

## 6. Add Translation Code

Paste the following code inside `indictrans2.py`:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"

print("Loading IndicTrans2 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("IndicTrans2 Ready")


def translate_text(text, src="eng_Latn", tgt="hin_Deva"):

    input_text = f"{src} {text}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=256,
            num_beams=5
        )

    output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return output[0]


if __name__ == "__main__":
    sentence = "Welcome to the PolyVerba project"
    print("Original:", sentence)
    print("Hindi:", translate_text(sentence))
7. First Run (Model Download)
Run:

python translation/indictrans2.py
First run will download the model (~1GB).
Wait until completion.

Expected output:

Original: Welcome to the PolyVerba project
Hindi: पॉलीवर्बा परियोजना में आपका स्वागत है
If you see this → installation successful.

8. Language Codes
Language	Code
English	eng_Latn
Hindi	hin_Deva
Kannada	kan_Knda
Tamil	tam_Taml
Telugu	tel_Telu
Marathi	mar_Deva
Bengali	ben_Beng
Example:
English → Kannada:

translate_text(text, "eng_Latn", "kan_Knda")
9. Connect to Whisper
Open:

stt/system_audio.py
Add at the top:

from translation.indictrans2 import translate_text
Find:

print(seg.text)
Replace with:

print("EN:", seg.text)

if len(seg.text.strip()) > 15:
    translated = translate_text(seg.text, "eng_Latn", "hin_Deva")
    print("HI:", translated)
10. Run the Full System
Start captioning:

python stt/system_audio.py
Play a video or meeting audio.

You should see:

EN: Today we will study neural networks
HI: आज हम न्यूरल नेटवर्क का अध्ययन करेंगे
11. Performance Optimization
Use 3‑second audio chunks in system_audio.py:

numframes = SAMPLE_RATE * 3
Reason:
IndicTrans2 works best on complete sentences, not short fragments.

12. Troubleshooting
Model not downloading
Check internet connection and rerun.

CUDA error
You do not need GPU. It automatically uses CPU.

Slow translation
Normal on first run. Later runs are faster.

No Hindi output
Make sure:

len(seg.text.strip()) > 15
is present.

13. What You Have Built
You now have a working bilingual captioning engine:

Laptop Audio
      ↓
Whisper Speech Recognition
      ↓
English Text
      ↓
IndicTrans2 Translation
      ↓
Hindi Captions
This is the core translation engine of PolyVerba.




---

