import sounddevice as sd
import numpy as np
import sys
import queue
import time
from stt.transcribe import Transcriber

# Configuration
BLOCK_DURATION = 0.2     # Audio chunk size (latency lower bound)
PARTIAL_UPDATE_INTERVAL = 0.5 # How often to update partial text
SILENCE_THRESHOLD = 0.01 # RMS threshold for silence
SILENCE_DURATION_TO_COMMIT = 1.0 # Seconds of silence to trigger commit

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

def main():
    # ... setup code ...
    
    phrase_buffer = []  # Accumulates current sentence
    silence_counter = 0 # Tracks consecutive silent chunks
    last_transcript_time = 0
    
    while True:
        # 1. Get Chunk
        chunk = audio_queue.get()
        phrase_buffer.append(chunk)
        
        # 2. Check Silence
        rms = np.sqrt(np.mean(chunk**2))
        if rms < SILENCE_THRESHOLD:
            silence_counter += BLOCK_DURATION
        else:
            silence_counter = 0
            
        # 3. Commit Decision (End of sentence)
        if silence_counter >= SILENCE_DURATION_TO_COMMIT and len(phrase_buffer) > 5:
             # Transcribe full phrase, print with \n, clear buffer
             pass
             
        # 4. Partial Update (Typing effect)
        elif time.time() - last_transcript_time > PARTIAL_UPDATE_INTERVAL:
             # Transcribe phrase_buffer, print with \r
             pass
