from faster_whisper import WhisperModel
import os

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
