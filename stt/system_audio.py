import sounddevice as sd
import numpy as np
import sys
import queue
import time
import threading
import torch
from stt.transcribe import Transcriber

try:
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
except ImportError:
    pass

class SpeakerManager:
    def __init__(self, similarity_threshold=0.78):
        self.sessions = {}
        self.classifier = None
        self.similarity_threshold = similarity_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.audio_buffers = {}
        
    def load_model(self):
        if self.classifier is None:
            print("Loading SpeechBrain ECAPA-TDNN...")
            import os
            os.environ["TQDM_DISABLE"] = "1"
            try:
                from speechbrain.inference.speaker import EncoderClassifier
                self.classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb", 
                    run_opts={"device": self.device},
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                print("Speaker model loaded successfully!")
            except Exception as e:
                print(f"Failed to load speaker classifier: {e}")
                self.classifier = False
                
    def process_chunk(self, audio_data, sample_rate=16000, session_id="default"):
        if self.classifier is None:
            self.load_model()
            
        if not self.classifier:
            return "Speaker 1", 1.0, {}, False
            
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "speakers": {},
                "speaker_count": 0,
                "history": [] # For temporal smoothing
            }
            
        session = self.sessions[session_id]
        
        # 1. Audio Quality Constraint
        rms = float(np.sqrt(np.mean(audio_data**2)))
        if rms < 0.005:
            last_spk = session["history"][-1] if session["history"] else "Speaker 1"
            return last_spk, 0.0, {}, False
            
        # Buffer to ensure minimum length (2-3 seconds)
        if session_id not in self.audio_buffers:
            self.audio_buffers[session_id] = np.array([], dtype=np.float32)
            
        self.audio_buffers[session_id] = np.concatenate([self.audio_buffers[session_id], audio_data])
        # Keep last 3 seconds
        max_len = 3 * sample_rate
        if len(self.audio_buffers[session_id]) > max_len:
            self.audio_buffers[session_id] = self.audio_buffers[session_id][-max_len:]
            
        embedding_audio = self.audio_buffers[session_id]
        
        if len(embedding_audio) < 1.5 * sample_rate:
            last_spk = session["history"][-1] if session["history"] else "Speaker 1"
            return last_spk, 0.0, {}, False
            
        # Normalize audio before embedding
        max_val = np.max(np.abs(embedding_audio))
        if max_val > 0:
            embedding_audio = embedding_audio / max_val
            
        try:
            signal = torch.from_numpy(embedding_audio).float().unsqueeze(0)
            with torch.no_grad():
                embeddings = self.classifier.encode_batch(signal)
                emb = embeddings.squeeze().cpu().numpy()
                
            if emb.ndim == 0:
                last_spk = session["history"][-1] if session["history"] else "Speaker 1"
                return last_spk, 0.0, {}, False
                
            # 2. Normalize embedding before comparison
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            
            # 3. Speaker Matching
            sim_scores = {}
            for spk_id, data in session["speakers"].items():
                sim = float(np.dot(emb, data["centroid"]))
                sim_scores[spk_id] = sim
                
            print(f"[DIARIZATION] All similarity scores: {sim_scores}")
            
            best_spk = None
            max_sim = -1.0
            if sim_scores:
                best_spk = max(sim_scores, key=sim_scores.get)
                max_sim = sim_scores[best_spk]
                
            is_new = False
            raw_pred_spk = None
            
            if best_spk is not None and max_sim >= self.similarity_threshold:
                raw_pred_spk = best_spk
                # 5. Centroid Update (mean of last 5)
                session["speakers"][best_spk]["embeddings"].append(emb)
                if len(session["speakers"][best_spk]["embeddings"]) > 5:
                    session["speakers"][best_spk]["embeddings"].pop(0)
                
                centroid = np.mean(session["speakers"][best_spk]["embeddings"], axis=0)
                session["speakers"][best_spk]["centroid"] = centroid / (np.linalg.norm(centroid) + 1e-8)
            else:
                # Create new speaker
                session["speaker_count"] += 1
                new_spk_id = f"Speaker {session['speaker_count']}"
                session["speakers"][new_spk_id] = {
                    "embeddings": [emb],
                    "centroid": emb
                }
                raw_pred_spk = new_spk_id
                is_new = True
                print(f"[DIARIZATION] New speaker created: {new_spk_id} (max similarity was {max_sim:.3f})")
                
            # 6. Temporal Smoothing (most frequent of last 3)
            session["history"].append(raw_pred_spk)
            if len(session["history"]) > 3:
                session["history"].pop(0)
                
            import collections
            counts = collections.Counter(session["history"])
            final_spk = counts.most_common(1)[0][0]
            
            print(f"[DIARIZATION] Selected speaker: {final_spk} (raw prediction: {raw_pred_spk})")
            
            if not hasattr(self, 'last_speaker'):
                self.last_speaker = None
            if self.last_speaker != final_spk and self.last_speaker is not None:
                print(f"[DIARIZATION] Speaker changed: {self.last_speaker} -> {final_spk}")
            self.last_speaker = final_spk
            
            return final_spk, max_sim, sim_scores, is_new
                
        except Exception as e:
            print(f"Embedding Error: {e}")
            last_spk = session["history"][-1] if session["history"] else "Speaker 1"
            return last_spk, 0.0, {}, False

speaker_manager = SpeakerManager(similarity_threshold=0.78)

def get_speaker_label(audio_data, sample_rate=16000):
    speaker, max_sim, sim_scores, is_new = speaker_manager.process_chunk(audio_data)
    return speaker

inference_lock = threading.Lock()

# Configuration
BLOCK_DURATION = 0.1     # Audio chunk size (latency lower bound)
PARTIAL_UPDATE_INTERVAL = 1.0 # Increased from 0.1 to 1.0 to prevent CPU overload and reduce latency
SILENCE_THRESHOLD = 0.005
SILENCE_DURATION_TO_COMMIT = 0.4 
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
        # Prefer MME first for maximum compatibility on Windows to avoid WdmSyncIoctl errors
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "MME" in api_name:
                print(f"Selected CABLE (MME): {i} - {dev['name']}")
                return i, dev

        # Fallback to DirectSound
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "DirectSound" in api_name:
                print(f"Selected CABLE (DS): {i} - {dev['name']}")
                return i, dev

        # Fallback to WASAPI
        for i, dev in cable_candidates:
            api_name = hostapis[dev['hostapi']]['name']
            if "WASAPI" in api_name:
                print(f"Selected CABLE (WASAPI): {i} - {dev['name']}")
                return i, dev
                
        # Last Resort Fallback to any CABLE
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

import io
import scipy.io.wavfile as wavfile
import requests
import os
from dotenv import load_dotenv

load_dotenv()

FLORES_TO_SARVAM = {
    "eng_Latn": "en-IN",
    "en": "en-IN",
    "hin_Deva": "hi-IN",
    "hi": "hi-IN",
    "tam_Taml": "ta-IN",
    "ta": "ta-IN",
    "tel_Telu": "te-IN",
    "te": "te-IN",
    "pan_Guru": "pa-IN",
    "pa": "pa-IN",
    "guj_Gujr": "gu-IN",
    "gu": "gu-IN",
    "mar_Deva": "mr-IN",
    "mr": "mr-IN",
    "ben_Beng": "bn-IN",
    "bn": "bn-IN",
    "mal_Mlym": "ml-IN",
    "ml": "ml-IN",
    "kan_Knda": "kn-IN",
    "kn": "kn-IN",
    "ory_Orya": "or-IN",
    "or": "or-IN",
}

def _execute_sarvam(api_key, audio_bytes, src, tgt, use_translation):
    headers = {"api-subscription-key": api_key}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": "saaras:v3", "language_code": src if src != "en-IN" else "hi-IN"} 
    
    start_time = time.time()
    print("[SARVAM] API request sent to https://api.sarvam.ai/speech-to-text")
    stt_res = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data, timeout=8)
    stt_latency = (time.time() - start_time) * 1000
    
    req_id = stt_res.headers.get("x-request-id", "unknown")
    print(f"[SARVAM] Response received. Status: {stt_res.status_code}, Latency: {stt_latency:.2f}ms, Request ID: {req_id}")

    if not stt_res.ok:
        raise Exception(f"STT Error: {stt_res.text}")
        
    transcript = stt_res.json().get("transcript", "")
    if not transcript or not transcript.strip():
        return ""
        
    if use_translation and src != tgt:
        trans_payload = {
            "input": transcript,
            "source_language_code": src,
            "target_language_code": tgt,
            "speaker_gender": "Male",
            "mode": "formal"
        }
        trans_headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
        start_time = time.time()
        print("[SARVAM] API request sent to https://api.sarvam.ai/translate")
        trans_res = requests.post("https://api.sarvam.ai/translate", headers=trans_headers, json=trans_payload, timeout=8)
        trans_latency = (time.time() - start_time) * 1000
        
        req_id_trans = trans_res.headers.get("x-request-id", "unknown")
        print(f"[SARVAM] Response received. Status: {trans_res.status_code}, Latency: {trans_latency:.2f}ms, Request ID: {req_id_trans}")
        
        if not trans_res.ok:
            raise Exception(f"Translate Error: {trans_res.text}")
        return trans_res.json().get("translated_text", "")
        
    return transcript

def process_sarvam_audio(audio_data, sample_rate, src_flores, tgt_flores, use_translation):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return "[Please configure SARVAM_API_KEY in .env]"
        
    src = FLORES_TO_SARVAM.get(src_flores, "hi-IN")
    tgt = FLORES_TO_SARVAM.get(tgt_flores, "en-IN")
    
    buffer = io.BytesIO()
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(buffer, sample_rate, audio_int16)
    audio_bytes = buffer.getvalue()
    
    max_retries = 2
    # Primary Edge Attempt
    for attempt in range(max_retries):
        try:
            return _execute_sarvam(api_key, audio_bytes, src, tgt, use_translation)
        except Exception as e:
            print(f"[SARVAM] Execution attempt failed: {repr(str(e))}")
            time.sleep(0.5)
            
    # Cloud Fallback Attempt
    print("[WARNING] Primary execution failed. Falling back to Cloud retry...")
    for attempt in range(max_retries):
        try:
            return _execute_sarvam(api_key, audio_bytes, src, tgt, use_translation)
        except Exception as e:
            print(f"[SARVAM] Fallback attempt failed: {repr(str(e))}")
            time.sleep(0.5)
            
    return "[Network Failure during Sarvam API execution]"

from stt.transcribe import Transcriber
from stt.translate import Translator

_simulate_network_failure = False

def set_network_failure(val):
    global _simulate_network_failure
    _simulate_network_failure = val

def transcription_worker(model_config, tx_queue, res_queue):
    """Worker thread that routes between Sarvam API and Local Edge based on routing_mode."""
    try:
        model_size = model_config.get("model", "tiny.en")
        use_translation = model_config.get("translate", False)
        source_lang = model_config.get("source_lang", "eng_Latn")
        target_lang = model_config.get("target_lang", "hin_Deva")
        routing_mode = model_config.get("routing_mode", "auto")
        
        # Load local models (For Edge/Fallback Execution)
        print(f"Loading Local Whisper model ({model_size})...")
        if source_lang != "eng_Latn" and ".en" in model_size:
            print("Warning: Source is not English, switching to 'base' model.")
            model_size = "base"
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        global _loaded_transcriber, _loaded_translator
        transcriber = None
        translator = None
        
        try:
            if '_loaded_transcriber' not in globals() or getattr(_loaded_transcriber, 'model_size', None) != model_size:
                _loaded_transcriber = Transcriber(model_size=model_size, device=device, compute_type="int8" if device=="cpu" else "float16")
                _loaded_transcriber.model_size = model_size
            transcriber = _loaded_transcriber
        except Exception as e:
            print(f"Failed to load Whisper: {e}")
            
        if use_translation:
            try:
                if '_loaded_translator' not in globals() or getattr(_loaded_translator, 'source_lang', None) != source_lang or getattr(_loaded_translator, 'target_lang', None) != target_lang:
                    _loaded_translator = Translator(target_lang=target_lang, source_lang=source_lang)
                translator = _loaded_translator
            except Exception as e:
                print(f"Failed to load Translator: {e}")
                
        print("Models loaded. Worker ready.")
        
        while True:
            audio_data, is_final = tx_queue.get()
            
            if audio_data is None: 
                break
                
            start_time = time.time()
                
            try:
                with inference_lock:
                    speaker = get_speaker_label(audio_data)
                    text = ""
                    fallback_used = False
                    sarvam_api_called = False
                    edge_processing = False
                    
                    sim_net_fail = _simulate_network_failure
                    
                    if routing_mode == "local_only":
                        fallback_used = True
                    elif routing_mode == "cloud_only":
                        if sim_net_fail:
                            text = ""
                            fallback_used = False # Forces failure without falling back
                        else:
                            sarvam_api_called = True
                            api_res = process_sarvam_audio(audio_data, SAMPLE_RATE, source_lang, target_lang, use_translation)
                            if api_res == "[Network Failure during Sarvam API execution]" or "[Please configure SARVAM_API_KEY" in api_res:
                                text = ""
                            else:
                                text = api_res
                            fallback_used = False
                    else: # auto mode
                        if sim_net_fail:
                            print("[EDGE] Simulated network failure triggered.")
                            fallback_used = True
                        else:
                            sarvam_api_called = True
                            api_res = process_sarvam_audio(audio_data, SAMPLE_RATE, source_lang, target_lang, use_translation)
                            
                            if api_res == "[Network Failure during Sarvam API execution]" or "[Please configure SARVAM_API_KEY" in api_res:
                                fallback_used = True
                            else:
                                text = api_res
                                
                    # 2. Fallback / Local Execution
                    if fallback_used:
                        print("[FALLBACK] Switching to offline Edge backend (Faster-Whisper)")
                        if transcriber:
                            try:
                                edge_processing = True
                                whisper_lang = "en" if "eng" in source_lang else ("hi" if "hin" in source_lang else None)
                                segments, _ = transcriber.model.transcribe(audio_data, beam_size=1, vad_filter=True, language=whisper_lang)
                                raw_text = " ".join([s.text for s in segments]).strip()
                                
                                if raw_text:
                                    if use_translation and translator:
                                        text = translator.translate(raw_text)
                                    else:
                                        text = raw_text
                            except Exception as e:
                                print(f"[Local Execution Failed] {repr(str(e))}")
                        else:
                            print("[FALLBACK] Local Edge model not loaded.")
                        
                    if text:
                        latency = time.time() - start_time
                        runtime_val = "edge" if fallback_used else "cloud"
                        execution_source_val = "edge_fallback" if fallback_used else "sarvam_api"
                        
                        res_queue.put({
                            "text": text,
                            "is_final": is_final,
                            "latency": latency,
                            "speaker": speaker,
                            "execution_source": execution_source_val,
                            "sarvam_api_called": sarvam_api_called,
                            "edge_processing": edge_processing,
                            "fallback_triggered": fallback_used,
                            "latency_ms": latency * 1000,
                            "speaker_detected": speaker,
                            "runtime": runtime_val
                        })
                        
            except Exception as e:
                print(f"Error in transcription/translation: {repr(str(e))}")
                
            tx_queue.task_done()
            
    except Exception as e:
        print(f"Worker failed: {e}")

# API Control Globals
current_stop_event = threading.Event()
capture_thread = None
transcription_thread = None

def start_transcription(source_lang, target_lang, model, translate=False, routing_mode="auto"):
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
        "target_lang": target_lang,
        "routing_mode": routing_mode
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
        with transcription_queue.mutex: transcription_queue.queue.clear()
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
    absolute_silence_time = 0
    warning_printed = False
    
    try:
        with sd.InputStream(device=device_id, channels=1, samplerate=device_samplerate, blocksize=block_size, callback=callback):
            while not stop_event.is_set():
                # 1. Process Audio Queue
                try:
                    while not audio_queue.empty():
                        chunk = audio_queue.get_nowait()
                        phrase_buffer.append(chunk)
                        
                        rms = np.sqrt(np.mean(chunk**2))
                        
                        # Tracking absolute silence (routing issues)
                        if rms == 0.0:
                            absolute_silence_time += BLOCK_DURATION
                            if absolute_silence_time > 5.0 and not warning_printed:
                                print("\n[WARNING] Receiving absolute silence. Ensure your Windows Playback device is set to 'CABLE Input' and audio is actively playing.")
                                warning_printed = True
                        else:
                            absolute_silence_time = 0
                            
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
    success = start_transcription(args.source_lang, args.target_lang, args.model, args.translate, False)
    if not success:
        return

    # CLI Loop to consume results and print them
    try:
        current_displayed_text = ""
        while True:
             try:
                while not result_queue.empty():
                    res_dict = result_queue.get_nowait()
                    if isinstance(res_dict, dict):
                        text = res_dict.get("text", "")
                        is_final = res_dict.get("is_final", False)
                        speaker = res_dict.get("speaker", "Speaker")
                    else:
                        text, is_final, latency, speaker = res_dict
                        
                    if text:
                        current_displayed_text = f"[{speaker}] {text}"
                        if is_final:
                            # Clear line first to be safe
                            cols = shutil.get_terminal_size().columns
                            sys.stdout.write("\r" + " " * (cols - 1) + "\r")
                            sys.stdout.write(f"[{speaker}] {text}\n")
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
