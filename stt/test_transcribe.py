
from transcribe import Transcriber
import numpy as np

print("Testing Transcriber initialization...")
try:
    # Use small model and CPU for quick test
    transcriber = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    print("Transcriber initialized.")
    
    # Create dummy audio data (1 second of silence)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    
    print("Testing transcription...")
    segments, info = transcriber.model.transcribe(dummy_audio, beam_size=1)
    
    print("Transcription successful (even if empty).")
    for segment in segments:
        print(f"Segment: {segment.text}")
        
except Exception as e:
    print(f"Test failed: {e}")
