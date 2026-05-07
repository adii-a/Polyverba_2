import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model definitions
MODELS = {
    "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",
    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model_direction(src, tgt):
    if src == "eng_Latn" and tgt != "eng_Latn":
        return "en-indic"
    elif src != "eng_Latn" and tgt == "eng_Latn":
        return "indic-en"
    elif src != "eng_Latn" and tgt != "eng_Latn":
        return "indic-indic"
    return None

def load_model(direction):
    if direction not in MODELS:
        raise ValueError(f"Invalid direction: {direction}")
        
    if direction in loaded_models:
        return loaded_models[direction]
        
    model_name = MODELS[direction]
    print(f"Loading IndicTrans2 Model: {direction} ({model_name})...")
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    
    # Optional: compile model for even faster inference if using PyTorch 2.0+
    # model = torch.compile(model)
    
    loaded_models[direction] = (tokenizer, model)
    return tokenizer, model

def translate_text(text, src="eng_Latn", tgt="hin_Deva"):
    if not text or not text.strip():
        return ""

    direction = get_model_direction(src, tgt)

    if not direction:
        print(f"Unsupported translation direction: {src} -> {tgt}")
        return text

    try:
        tokenizer, model = load_model(direction)
        
        # Determine prefix for specific models if needed (not needed for v2, just inputs)
        # IndicTrans2 v2 uses standard mBART-like format
        
        input_text = f"{src} {tgt} {text}"
        
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
                num_beams=1
            )

        output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        return output[0]
        
    except Exception as e:
        print(f"Translation failed: {e}")
        return text

if __name__ == "__main__":
    sentence = "Welcome to the PolyVerba project"
    print("Original:", sentence)
    print("Hindi:", translate_text(sentence, src="eng_Latn", tgt="hin_Deva"))

