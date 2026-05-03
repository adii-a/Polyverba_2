import sounddevice as sd
import numpy as np
import sys
import queue
import time
import threading
import torch
from stt.transcribe import Transcriber

# Configuration
BLOCK_DURATION = 0.2     # Audio chunk size (latency lower bound)
PARTIAL_UPDATE_INTERVAL = 0.5 # How often to update text
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION_TO_COMMIT = 0.6 
SAMPLE_RATE = 16000     

audio_queue = queue.Queue()
transcription_queue = queue.Queue() # For passing audio to worker thread
result_queue = queue.Queue()        # For passing text back to main thread

def find_cable_output():
    """Finds the audio input device."""
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    
    # 1. Try to find CABLE Output
    cable_candidates = []
    for i, dev in enumerate(devices):
        if "CABLE Output" in dev['name'] and dev['max_input_channels'] > 0:
            cable_candidates.append((i, dev))
            
    if cable_candidates:
        # Prefer MME or DirectSound for compatibility (WASAPI/WDM-KS can be flaky)
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "MME" in api_name:
                print(f"Selected CABLE (MME): {i} - {dev['name']}")
                return i, dev
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "DirectSound" in api_name:
                print(f"Selected CABLE (DS): {i} - {dev['name']}")
                return i, dev
                
        # Fallback to any CABLE
        i, dev = cable_candidates[0]
        print(f"Selected CABLE (Other): {i} - {dev['name']}")
        return i, dev

    # 2. Fallback to Default Input
    print("CABLE Output not found. Using System Default Input.")
    try:
        default_in = sd.default.device[0]
        dev_info = devices[default_in]
        print(f"Selected Default: {default_in} - {dev_info['name']}")
        return default_in, dev_info
    except Exception as e:
        print(f"Could not get default device: {e}")
        return None, None

def callback(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

from stt.translate import Translator, FLORES_CODES

import argparse
import shutil

def transcription_worker(model_config, tx_queue, res_queue):
    """Worker thread that runs the heavy Whisper inference AND Translation."""
    try:
        model_size = model_config.get("model", "tiny.en")
        use_translation = model_config.get("translate", False)
        source_lang = model_config.get("source_lang", "en")
        target_lang = model_config.get("target_lang", "hi")
        
        # Load model in thread
        print(f"Loading Whisper model ({model_size})...")
        # If source is not English, we must use a multilingual model if not already specified
        if source_lang != "en" and ".en" in model_size:
            print(f"Warning: Source is {source_lang}, but model is {model_size}. Switching to 'base' for multilingual support.")
            model_size = "base"
            
        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        from dotenv import load_dotenv
        load_dotenv()
        
        global _loaded_transcriber, _loaded_translator, _cloud_transcriber, _cloud_translator
        if '_loaded_transcriber' not in globals() or getattr(_loaded_transcriber, 'model_size', None) != model_size:
            print(f"Loading Whisper on device: {device}")
            _loaded_transcriber = Transcriber(model_size=model_size, device=device, compute_type="int8" if device=="cpu" else "float16")
            _loaded_transcriber.model_size = model_size
        transcriber = _loaded_transcriber
        
        from stt.transcribe import CloudWhisperEngine
        if '_cloud_transcriber' not in globals():
            _cloud_transcriber = CloudWhisperEngine()
        cloud_transcriber = _cloud_transcriber
        
        translator = None
        cloud_translator = None
        if use_translation:
            if '_loaded_translator' not in globals() or getattr(_loaded_translator, 'source_lang', None) != source_lang or getattr(_loaded_translator, 'target_lang', None) != target_lang:
                print(f"Loading Translator (Source: {source_lang} -> Target: {target_lang})...")
                _loaded_translator = Translator(target_lang=target_lang, source_lang=source_lang)
            translator = _loaded_translator
            
            from stt.translate import CloudTranslateEngine
            if '_cloud_translator' not in globals() or getattr(_cloud_translator, 'source_lang', None) != source_lang or getattr(_cloud_translator, 'target_lang', None) != target_lang:
                _cloud_translator = CloudTranslateEngine(target_lang=target_lang, source_lang=source_lang)
            cloud_translator = _cloud_translator
        
        print("Models loaded. Worker ready.")
        
        # State to cache detected language over consecutive streaming chunks
        current_detected_lang = None
        
        while True:
            # Get audio buffer
            audio_data, is_final = tx_queue.get()
            
            if audio_data is None: # Sentinel
                break
                
            # Transcribe
            try:
                # Decide which language to pass to whisper
                if source_lang != "auto":
                    whisper_lang = source_lang
                else:
                    # Use cached detection if we have one, otherwise let it auto-detect on this pass
                    whisper_lang = current_detected_lang
                    
                try:
                    segments, info = transcriber.model.transcribe(
                        audio_data, 
                        beam_size=1, 
                        vad_filter=True, 
                        language=whisper_lang,
                        condition_on_previous_text=False
                    )
                    source_identifier = "Local"
                except Exception as e:
                    print(f"Local STT Failed: {e}. Falling back to Cloud...")
                    segments, info = cloud_transcriber.transcribe(audio_data, whisper_lang)
                    source_identifier = "Cloud"
                
                # If doing auto-detect, lock in the language if we are highly confident
                if source_lang == "auto" and whisper_lang is None:
                    if hasattr(info, 'language_probability') and getattr(info, 'language_probability', 0) > 0.65:
                        current_detected_lang = info.language
                        print(f"Auto-detected language locked for phrase: {current_detected_lang} ({info.language_probability})")
                
                text = " ".join([s.text for s in segments]).strip()
                
                # Reset detection lock at the end of phrase, ready for a new speaker/language
                if is_final:
                    current_detected_lang = None
                
                if text:
                    if is_final:
                        if translator:
                            try:
                                translated_text = translator.translate(text)
                            except Exception as e:
                                print(f"Local Translate Failed ({e}). Falling back to Cloud...")
                                translated_text = cloud_translator.translate(text)
                                source_identifier = "Cloud"
                                
                            res_queue.put((translated_text, is_final, source_identifier))
                        else:
                            res_queue.put((text, is_final, source_identifier))
                    else:
                        # Partial update - LIVE TRANSLATION ENABLED
                        if translator:
                            try:
                                partial_translation = translator.translate(text)
                                res_queue.put((partial_translation, is_final, source_identifier))
                            except Exception:
                                res_queue.put((text, is_final, source_identifier))
                        else:
                            res_queue.put((text, is_final, source_identifier))
            except Exception as e:
                print(f"Error in transcription/translation: {e}")
                
            tx_queue.task_done()
            
    except Exception as e:
        print(f"Worker failed: {e}")

# API Control Globals
current_stop_event = threading.Event()
capture_thread = None
transcription_thread = None

def start_transcription(source_lang, target_lang, model, translate=False):
    """Starts the audio capture and transcription in a separate thread."""
    global capture_thread, transcription_thread, current_stop_event, transcription_queue, result_queue
    
    # Ensure any previous threads are really dead (or at least stopped)
    if (capture_thread and capture_thread.is_alive()) or (transcription_thread and transcription_thread.is_alive()):
        print("Stopping existing threads before starting new ones...")
        stop_transcription()
        
        # Wait for threads to actually release resources, with a short timeout to prevent hanging
        if capture_thread and capture_thread.is_alive():
            capture_thread.join(timeout=2.0)
        if transcription_thread and transcription_thread.is_alive():
            transcription_queue.put((None, None)) # Ensure worker gets the stop signal
            transcription_thread.join(timeout=2.0)

    # Create a BRAND NEW event so we don't accidentally clear the event old threads are still checking
    current_stop_event = threading.Event()
    
    transcription_queue = queue.Queue()
    result_queue = queue.Queue()
    
    with audio_queue.mutex: audio_queue.queue.clear()

    # Start Worker Thread
    worker_config = {
        "translate": translate, 
        "model": model,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    transcription_thread = threading.Thread(target=transcription_worker, args=(worker_config, transcription_queue, result_queue), daemon=True)
    transcription_thread.start()

    # Find device
    device_id, device_info = find_cable_output()
    if device_id is None:
        print("Error: No valid audio input device found.")
        current_stop_event.set()
        transcription_queue.put((None, None))
        return False

    device_samplerate = int(device_info['default_samplerate'])
    
    capture_thread = threading.Thread(target=audio_capture_loop, args=(device_id, device_samplerate, current_stop_event), daemon=True)
    capture_thread.start()
    
    return True

def stop_transcription():
    """Stops the audio capture and transcription."""
    global current_stop_event, capture_thread, transcription_thread, transcription_queue
    # Signal threads to stop via their bound event object
    if current_stop_event:
        current_stop_event.set()
    
    # Signal worker to stop
    if 'transcription_queue' in globals():
        transcription_queue.put((None, None))
    
    print("Threads signaled to stop.")

def audio_capture_loop(device_id, device_samplerate, stop_event):
    """Refactored main loop for audio capture."""
    block_size = int(device_samplerate * BLOCK_DURATION)
    
    # Resampling step
    step = 1
    if device_samplerate != SAMPLE_RATE:
        step = int(device_samplerate / SAMPLE_RATE)

    print(f"Listening on device ID: {device_id} at {device_samplerate}Hz")
    
    phrase_buffer = []  
    silence_counter = 0 
    last_transcript_request_time = 0
    
    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=device_samplerate, blocksize=block_size, callback=callback):
            while not stop_event.is_set():
                # 1. Process Audio Queue
                try:
                    while not audio_queue.empty():
                        chunk = audio_queue.get_nowait()
                        phrase_buffer.append(chunk)
                        
                        rms = np.sqrt(np.mean(chunk**2))
                        if rms < SILENCE_THRESHOLD:
                            silence_counter += BLOCK_DURATION
                        else:
                            silence_counter = 0
                except queue.Empty:
                    pass
                
                # 2. Check for Transcription Results (UI will poll result_queue directly, or we can print to stdout here if needed)
                # For CLI usage, we print. For API, we leave it in queue or consume it elsewhere.
                # To support both, we can peek/get here and put it back? No, result_queue is consumed by consumer.
                # For CLI mode, the main() function consumes it. For API, the API consumes it.
                # BUT this loop is running in a thread now.
                # Let's keep this loop FOCUSED on Audio -> Transcription Queue.
                # The Result Queue consumption should be separate.
                pass 

                # 3. Schedule Transcription
                current_time = time.time()
                buffer_duration = len(phrase_buffer) * BLOCK_DURATION
                
                should_commit = (silence_counter >= SILENCE_DURATION_TO_COMMIT and buffer_duration > 1.0) or (buffer_duration > 15.0)
                should_update = (current_time - last_transcript_request_time > PARTIAL_UPDATE_INTERVAL) and buffer_duration > 0.5
                
                if (should_commit or should_update) and transcription_queue.empty():
                    if not phrase_buffer:
                        continue
                        
                    audio_data = np.concatenate(phrase_buffer, axis=0)
                    audio_data = audio_data.flatten()
                    if step > 1:
                        audio_data = audio_data[::step]
                    
                    transcription_queue.put((audio_data, should_commit))
                    last_transcript_request_time = current_time
                    
                    if should_commit:
                        phrase_buffer = []
                        silence_counter = 0

                time.sleep(0.01)
                
    except Exception as e:
        print(f"Capture Loop Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="PolyVerba System Audio Captions")
    parser.add_argument("--translate", action="store_true", help="Enable translation")
    parser.add_argument("--model", type=str, default="base.en", help="Whisper model size")
    parser.add_argument("--source_lang", type=str, default="en", help="Source language code (e.g. en, hi, ta)")
    parser.add_argument("--target_lang", type=str, default="hi", help="Target language code (e.g. hi, en, ta)")
    args = parser.parse_args()

    # Auto-enable translation if languages differ
    if args.source_lang != args.target_lang:
        args.translate = True

    # Interactive Language Selection if not specified and translation enabled
    if args.translate and args.target_lang == "hi" and not any(arg in sys.argv for arg in ["--target_lang"]):
        print("\n--- Select Target Language ---")
        languages = list(FLORES_CODES.keys())
        display_langs = [l for l in languages if l != args.source_lang]
        
        for i, lang in enumerate(display_langs):
            print(f"{i+1}. {lang}")
            
        try:
            choice = input(f"\nEnter number (default=Hindi): ")
            if choice.strip():
                idx = int(choice) - 1
                if 0 <= idx < len(display_langs):
                    args.target_lang = display_langs[idx]
                    print(f"Selected: {args.target_lang}")
        except Exception:
            print("Invalid selection, defaulting to Hindi (hi)")

    print("Initializing Threaded Streaming Capture (CLI Mode)...")
    
    # Start the engine via API function
    success = start_transcription(args.source_lang, args.target_lang, args.model, args.translate)
    if not success:
        return

    # CLI Loop to consume results and print them
    try:
        current_displayed_text = ""
        while True:
             try:
                while not result_queue.empty():
                    result = result_queue.get_nowait()
                    text = result[0]
                    is_final = result[1]
                    src_id = result[2] if len(result) > 2 else "Local"
                    if text:
                        current_displayed_text = text
                        if is_final:
                            # Clear line first to be safe
                            cols = shutil.get_terminal_size().columns
                            sys.stdout.write("\r" + " " * (cols - 1) + "\r")
                            sys.stdout.write(f"{text}\n")
                            current_displayed_text = ""
                        else:
                            # Truncate to terminal width to avoid wrapping/repetition
                            cols = shutil.get_terminal_size().columns
                            max_len = cols - 5
                            display_text = text
                            if len(display_text) > max_len:
                                display_text = "..." + display_text[-(max_len-3):]
                                
                            # Pad with spaces to clear previous chars
                            sys.stdout.write(f"\r{display_text}")
                    sys.stdout.flush()
                time.sleep(0.05)
             except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_transcription()

if __name__ == "__main__":
    main()
