from faster_whisper import WhisperModel
import os
import base64
import requests

class Transcriber:
    def __init__(self, model_size="small", device="auto", compute_type="int8"):
        """
        Initializes the Whisper model.
        device: 'cuda' or 'cpu' (or 'auto')
        compute_type: 'int8', 'float16', 'float32'
        """
        print(f"Loading Whisper model: {model_size} on {device}...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU int8...")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_array):
        """
        Transcribes the given audio array (numpy array).
        Returns a generator of segments.
        """
        segments, info = self.model.transcribe(audio_array, beam_size=1, vad_filter=True)
        return segments, info

class CloudWhisperEngine:
    def __init__(self):
        self.endpoint_url = os.environ.get("MODAL_TRANSCRIBE_URL")

    def transcribe(self, audio_array, whisper_lang=None):
        if not self.endpoint_url:
            raise ValueError("MODAL_TRANSCRIBE_URL not set in environment.")

        audio_bytes = audio_array.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        payload = {
            "audio_b64": audio_b64,
            "source_lang": whisper_lang if whisper_lang else "auto"
        }
        
        print(f"[Cloud Fallback] Sending audio to Modal Whisper...")
        response = requests.post(self.endpoint_url, json=payload, timeout=6.0)
        response.raise_for_status()
        text = response.json().get("text", "")
        
        class MockSegment:
            def __init__(self, t):
                self.text = t
        return [MockSegment(text)], type("MockInfo", (), {"language": whisper_lang})()
