import modal
import base64
import numpy as np
from fastapi import Request

app = modal.App("polyverba-cloud-fallback")

# Define the cloud environment
polyverba_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", 
        "transformers==4.37.2", 
        "faster-whisper", 
        "sacremoses", 
        "indic-nlp-library", 
        "accelerate",
        "numpy"
    )
)

@app.cls(gpu="T4", image=polyverba_image, container_idle_timeout=300)
class CloudPolyverba:
    @modal.enter()
    def setup(self):
        print("Loading Models onto Cloud GPU...")
        from faster_whisper import WhisperModel
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        # Load local STT equivalent (Base Multilingual)
        print("Loading Whisper...")
        self.stt_model = WhisperModel("base", device="cuda", compute_type="float16")
        
        # Load Translation Engine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator_models = {}
        
        # We load the primary direction (English -> Indic) to save VRAM and boot time
        direction = "en-indic"
        model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        print(f"Loading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
        model.to(self.device)
        self.translator_models[direction] = (tokenizer, model)
            
        print("Cloud Models Initialized!")

    @modal.web_endpoint(method="POST")
    async def transcribe(self, request: Request):
        """Receives an audio array and returns the transcription."""
        data = await request.json()
        audio_b64 = data.get("audio_b64")
        source_lang = data.get("source_lang", "en")
        
        if not audio_b64:
            return {"text": ""}
            
        audio_bytes = base64.b64decode(audio_b64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        language_arg = source_lang if source_lang != "auto" else None
        segments, info = self.stt_model.transcribe(audio_array, beam_size=1, vad_filter=True, language=language_arg, condition_on_previous_text=False)
        text = " ".join([s.text for s in segments]).strip()
        
        return {"text": text}

    @modal.web_endpoint(method="POST")
    async def translate(self, request: Request):
        """Translates english text to a target indic language."""
        data = await request.json()
        text = data.get("text", "")
        src = data.get("src", "eng_Latn")
        tgt = data.get("tgt", "hin_Deva")
        
        if not text:
            return {"text": ""}
            
        direction = "en-indic"
        if direction not in self.translator_models:
            return {"text": text}
            
        tokenizer, model = self.translator_models[direction]
        input_text = f"{src} {tgt} {text}"
        
        import torch
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                max_length=256,
                num_beams=1
            )
            
        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return {"text": output[0]}
