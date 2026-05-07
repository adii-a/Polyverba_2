from deep_translator import GoogleTranslator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Map common codes to FLORES codes used by IndicTrans2
FLORES_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "ml": "mal_Mlym",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "pa": "pan_Guru",
    "ur": "urd_Arab",
    "or": "ory_Orya",
    "as": "asm_Beng"
}

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "bn": "Bengali",
    "pa": "Punjabi",
    "ur": "Urdu",
    "or": "Odia",
    "as": "Assamese"
}

class Translator:
    def __init__(self, target_lang='hi', source_lang='en', use_indictrans=True):
        """
        Initializes the translator.
        Args:
            target_lang: Target language code (2-letter).
            source_lang: Source language code (2-letter).
            use_indictrans: If True, tries to use IndicTrans2 for Indian languages.
        """
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.use_indictrans = use_indictrans
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check if pair is supported by IndicTrans2
        self.is_indic_pair = (source_lang in FLORES_CODES and target_lang in FLORES_CODES) and \
                             (source_lang != "en" or target_lang != "en")

        if self.use_indictrans and self.is_indic_pair:
            self._load_indictrans()
        else:
            self.translator = GoogleTranslator(source=source_lang, target=target_lang)
            print(f"Translator initialized (Deep-Translator): {source_lang} -> {target_lang}")

    def _load_indictrans(self):
        try:
            print("Loading IndicTrans2 Module...")
            # Lazy import to trigger model loading from the module
            from translation.indictrans2 import translate_text, load_model, get_model_direction
            self.indic_translate_func = translate_text
            
            # Eagerly load the model! 
            src_code = FLORES_CODES.get(self.source_lang, "eng_Latn")
            tgt_code = FLORES_CODES.get(self.target_lang, "hin_Deva")
            direction = get_model_direction(src_code, tgt_code)
            
            if direction:
                print(f"Pre-warming model for direction {direction}...")
                load_model(direction)
                
            print(f"IndicTrans2 Loaded via module.")
            
        except Exception as e:
            print(f"Failed to load IndicTrans2: {e}")
            print("Falling back to Google Translator...")
            self.use_indictrans = False
            self.translator = GoogleTranslator(source=self.source_lang, target=self.target_lang)

    def translate(self, text):
        """
        Translates the given text to the target language.
        """
        if not text or not text.strip():
            return ""
            
        if self.use_indictrans and hasattr(self, 'indic_translate_func'):
            try:
                src_code = FLORES_CODES.get(self.source_lang, "eng_Latn")
                tgt_code = FLORES_CODES.get(self.target_lang, "hin_Deva")
                
                with open("debug_server.log", "a") as f:
                    f.write(f"[DEBUG] FLORES keys: {list(FLORES_CODES.keys())[:5]}... ta in keys: {'ta' in FLORES_CODES}\n")
                    f.write(f"[DEBUG] IndicTrans call: src={src_code} ({self.source_lang}), tgt={tgt_code} ({self.target_lang}), text='{text[:20]}...'\n")

                return self.indic_translate_func(text, src=src_code, tgt=tgt_code)
                
            except Exception as e:
                print(f"IndicTrans2 Error: {e}")
                return text
        else:
            # Fallback
            try:
                return self.translator.translate(text)
            except Exception as e:
                print(f"Translation Error: {e}")
                return text
