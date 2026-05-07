import sounddevice as sd
import numpy as np
import sys
import queue
import time
import threading
import math
import torch
from scipy.signal import resample_poly
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

# Module-level model cache — keyed by (model, device, lang) so reloads only when needed
_model_cache: dict = {}

# True once the Whisper model has been loaded and is ready to transcribe
model_ready: bool = False

def preload_whisper(model_size: str = "base") -> None:
    """
    Pre-loads the Whisper model into _model_cache at server startup so that
    clicking Start has zero model-load delay.  Called from a background thread
    by web_server.py during the lifespan startup phase.
    """
    global model_ready
    try:
        from dotenv import load_dotenv
        load_dotenv()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        t_key = f"transcriber_{model_size}_{device}"
        if t_key not in _model_cache:
            print(f"[Preload] Loading Whisper ({model_size}) on {device}...")
            _model_cache[t_key] = Transcriber(
                model_size=model_size,
                device=device,
                compute_type="int8" if device == "cpu" else "float16"
            )
            print(f"[Preload] Whisper ({model_size}) ready ✓")
        else:
            print(f"[Preload] Whisper ({model_size}) already cached ✓")

        # Also pre-load the cloud fallback (lightweight, just sets up the client)
        from stt.transcribe import CloudWhisperEngine
        if 'cloud_transcriber' not in _model_cache:
            _model_cache['cloud_transcriber'] = CloudWhisperEngine()

        model_ready = True
    except Exception as e:
        print(f"[Preload] Failed to preload Whisper: {e}")
        # model_ready stays False — /api/start will surface the error to the UI


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
        # Prefer DirectSound for maximum compatibility with arbitrary block sizes
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "DirectSound" in api_name:
                print(f"Selected CABLE (DS): {i} - {dev['name']}")
                return i, dev
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "WASAPI" in api_name:
                print(f"Selected CABLE (WASAPI): {i} - {dev['name']}")
                return i, dev
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "MME" in api_name:
                print(f"Selected CABLE (MME): {i} - {dev['name']}")
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
    
    # Downmix to mono if multiple channels are captured
    if indata.ndim > 1 and indata.shape[1] > 1:
        mono_data = np.mean(indata, axis=1, keepdims=True)
        audio_queue.put(mono_data.copy())
    else:
        audio_queue.put(indata.copy())

from stt.translate import Translator, FLORES_CODES

import argparse
import shutil

import re as _re

def _translate_preserving_speaker_tags(text: str, translate_fn) -> str:
    """
    Strip [Speaker N]: prefix tags from `text`, translate only the speech
    content, then re-attach the original tags.  This ensures the frontend
    speaker-pill highlighting works correctly for every output language.

    Handles multi-speaker segments separated by newlines, e.g.:
        [Speaker 1]: Hello
        [Speaker 2]: World
    """
    # Pattern matches optional leading newline + [Speaker N]: at line start
    SPEAKER_RE = _re.compile(r'(\n?\[Speaker \d+\]:)')
    parts = SPEAKER_RE.split(text)
    # parts alternates: [plain?, tag, plain, tag, plain, ...]
    result = []
    for part in parts:
        if SPEAKER_RE.match(part):
            # It's a speaker tag — keep as-is
            result.append(part)
        elif part.strip():
            # It's speech content — translate it
            result.append(translate_fn(part))
        else:
            result.append(part)
    return "".join(result)

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
        
        t_key = f"transcriber_{model_size}_{device}"
        if t_key not in _model_cache:
            print(f"Loading Whisper ({model_size}) on device: {device}")
            _model_cache[t_key] = Transcriber(model_size=model_size, device=device, compute_type="int8" if device=="cpu" else "float16")
        transcriber = _model_cache[t_key]

        from stt.transcribe import CloudWhisperEngine
        if 'cloud_transcriber' not in _model_cache:
            _model_cache['cloud_transcriber'] = CloudWhisperEngine()
        cloud_transcriber = _model_cache['cloud_transcriber']
        
        translator = None
        cloud_translator = None
        diarizer = None
        if use_translation:
            tx_key = f"translator_{source_lang}_{target_lang}"
            if tx_key not in _model_cache:
                print(f"Loading Translator ({source_lang} -> {target_lang})...")
                _model_cache[tx_key] = Translator(target_lang=target_lang, source_lang=source_lang)
            translator = _model_cache[tx_key]

            from stt.translate import CloudTranslateEngine
            ctx_key = f"cloud_translator_{source_lang}_{target_lang}"
            if ctx_key not in _model_cache:
                _model_cache[ctx_key] = CloudTranslateEngine(target_lang=target_lang, source_lang=source_lang)
            cloud_translator = _model_cache[ctx_key]

            from stt.diarization import PyannoteDiarizer
            if 'diarizer' not in _model_cache:
                _model_cache['diarizer'] = PyannoteDiarizer()
            diarizer = _model_cache['diarizer']
        
        print("Models loaded. Worker ready.")
        
        # State to cache detected language over consecutive streaming chunks
        current_detected_lang = None
        # Persistent speaker label map — keeps labels consistent across the whole session
        speaker_label_map = {}
        def get_friendly_speaker(raw_id):
            if raw_id not in speaker_label_map:
                speaker_label_map[raw_id] = f"Speaker {len(speaker_label_map) + 1}"
            return speaker_label_map[raw_id]
        
        while True:
            # Get audio buffer
            audio_data, is_final = tx_queue.get()
            
            if audio_data is None: # Sentinel
                break
                
            # Transcribe
            start_time = time.time()
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
                        condition_on_previous_text=False,
                        word_timestamps=True
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
                
                segments_list = list(segments)
                if is_final and diarizer:
                    # Diarization is expensive — only run on committed final phrases
                    diarization_segments = diarizer.diarize(audio_data, SAMPLE_RATE)
                    formatted_text = ""
                    current_speaker = None
                    for segment in segments_list:
                        if not getattr(segment, 'words', None):
                            formatted_text += segment.text + " "
                            continue
                        for word in segment.words:
                            word_start = word.start
                            word_end = word.end
                            raw_speaker = "SPEAKER_00"
                            max_overlap = 0
                            for ds in diarization_segments:
                                overlap = max(0, min(word_end, ds['end']) - max(word_start, ds['start']))
                                if overlap > max_overlap:
                                    max_overlap = overlap
                                    raw_speaker = ds['speaker']
                            assigned_speaker = get_friendly_speaker(raw_speaker)
                            if assigned_speaker != current_speaker:
                                current_speaker = assigned_speaker
                                formatted_text += f"\n[{current_speaker}]:"
                            formatted_text += word.word
                    text = formatted_text.strip()
                else:
                    # Partial update: fast path — no diarization overhead
                    text = " ".join(seg.text for seg in segments_list).strip()
                
                # Reset detection lock at the end of phrase, ready for a new speaker/language
                if is_final:
                    current_detected_lang = None
                
                if text:
                    latency = round(time.time() - start_time, 2)
                    if is_final:
                        if translator:
                            try:
                                translated_text = _translate_preserving_speaker_tags(text, translator.translate)
                            except Exception as e:
                                print(f"Local Translate Failed ({e}). Falling back to Cloud...")
                                translated_text = _translate_preserving_speaker_tags(text, cloud_translator.translate)
                                source_identifier = "Cloud"
                                
                            res_queue.put((translated_text, is_final, source_identifier, latency))
                        else:
                            res_queue.put((text, is_final, source_identifier, latency))
                    else:
                        # Partial update - LIVE TRANSLATION ENABLED
                        if translator:
                            try:
                                partial_translation = _translate_preserving_speaker_tags(text, translator.translate)
                                res_queue.put((partial_translation, is_final, source_identifier, latency))
                            except Exception:
                                res_queue.put((text, is_final, source_identifier, latency))
                        else:
                            res_queue.put((text, is_final, source_identifier, latency))
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
    
    # Resampling ratio (used later when audio is processed)

    print(f"Listening on device ID: {device_id} at {device_samplerate}Hz")
    
    phrase_buffer = []  
    silence_counter = 0 
    last_transcript_request_time = 0
    
    try:
        # Let PortAudio use default channels and blocksize to maximize compatibility
        with sd.InputStream(device=device_id, samplerate=device_samplerate, callback=callback):
            while not stop_event.is_set():
                # 1. Process Audio Queue
                try:
                    while not audio_queue.empty():
                        chunk = audio_queue.get_nowait()
                        phrase_buffer.append(chunk)
                        
                        chunk_duration = len(chunk) / device_samplerate
                        rms = np.sqrt(np.mean(chunk**2))
                        if rms < SILENCE_THRESHOLD:
                            silence_counter += chunk_duration
                        else:
                            silence_counter = 0
                except queue.Empty:
                    pass
                
                # 2. Check for Transcription Results
                pass 

                # 3. Schedule Transcription
                current_time = time.time()
                buffer_duration = sum(len(c) for c in phrase_buffer) / device_samplerate if phrase_buffer else 0
                
                should_commit = (silence_counter >= SILENCE_DURATION_TO_COMMIT and buffer_duration > 1.0) or (buffer_duration > 15.0)
                should_update = (current_time - last_transcript_request_time > PARTIAL_UPDATE_INTERVAL) and buffer_duration > 0.5
                
                if (should_commit or should_update) and transcription_queue.empty():
                    if not phrase_buffer:
                        continue
                        
                    audio_data = np.concatenate(phrase_buffer, axis=0)
                    audio_data = audio_data.flatten()
                    if device_samplerate != SAMPLE_RATE:
                        g = math.gcd(SAMPLE_RATE, device_samplerate)
                        audio_data = resample_poly(audio_data, SAMPLE_RATE // g, device_samplerate // g)

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
