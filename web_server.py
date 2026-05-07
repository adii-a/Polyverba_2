from fastapi import FastAPI, WebSocket, Request, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import requests
import os
from dotenv import load_dotenv

load_dotenv()
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import queue
import time
import stt.system_audio as system_audio
from stt.translate import FLORES_CODES, LANGUAGE_NAMES

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

templates = Jinja2Templates(directory="web/templates")

# Store connected websockets
connected_clients = set()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "languages": LANGUAGE_NAMES,
            "v": int(time.time())
        }
    )

def process_sarvam_pipeline(audio_bytes: bytes, target_lang: str, fallback_used: bool):
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise Exception("SARVAM_API_KEY is missing in .env")

    # STT API
    headers = {"api-subscription-key": api_key}
    files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
    data = {"model": "saaras:v1", "language_code": "hi-IN"}
    
    stt_res = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, files=files, data=data)
    if not stt_res.ok:
        raise Exception(f"STT API Failed: {stt_res.text}")
        
    transcript = stt_res.json().get("transcript", "")
    
    # Translate API
    trans_payload = {
        "input": transcript,
        "source_language_code": "hi-IN",
        "target_language_code": target_lang,
        "speaker_gender": "Male",
        "mode": "formal"
    }
    trans_headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    trans_res = requests.post("https://api.sarvam.ai/translate", headers=trans_headers, json=trans_payload)
    
    if not trans_res.ok:
        raise Exception(f"Translate API Failed: {trans_res.text}")
        
    translated_text = trans_res.json().get("translated_text", "")
    
    return {
        "original_text": transcript,
        "translated_text": translated_text,
        "language_detected": "hi-IN",
        "fallback_used": fallback_used
    }

@app.get("/sarvam", response_class=HTMLResponse)
async def get_sarvam_page(request: Request):
    return templates.TemplateResponse(request=request, name="sarvam.html")

@app.post("/api/sarvam/edge")
async def sarvam_edge(audio: UploadFile = File(...), targetLanguage: str = Form(...)):
    try:
        audio_bytes = await audio.read()
        return process_sarvam_pipeline(audio_bytes, targetLanguage, False)
    except Exception as e:
        print(f"[Edge Error] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/sarvam/cloud")
async def sarvam_cloud(audio: UploadFile = File(...), targetLanguage: str = Form(...)):
    try:
        audio_bytes = await audio.read()
        return process_sarvam_pipeline(audio_bytes, targetLanguage, True)
    except Exception as e:
        print(f"[Cloud Error] {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/start")
async def start_transcription(request: Request, data: dict):
    print(f"[DEBUG] /api/start payload: {data}")
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "hi")
    model = data.get("model", "base.en")
    routing_mode = data.get("routing_mode", "auto")
    
    # Auto-translate logic
    translate = (source_lang != target_lang)
    
    success = system_audio.start_transcription(source_lang, target_lang, model, translate, routing_mode)
    if success:
        return {"status": "started", "message": f"Started {source_lang}->{target_lang}"}
    else:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to start"})

@app.post("/api/stop")
async def stop_transcription():
    system_audio.stop_transcription()
    return {"status": "stopped"}

@app.post("/api/simulate")
async def simulate_event(data: dict):
    if "network_failure" in data:
        system_audio.set_network_failure(data["network_failure"])
    return {"status": "ok"}

@app.get("/debug/status")
async def debug_status():
    session = system_audio.speaker_manager.sessions.get("default", {})
    speakers = list(session.get("speakers", {}).keys()) if session else []
    model_loaded = hasattr(system_audio, '_loaded_transcriber') and getattr(system_audio, '_loaded_transcriber') is not None
    
    return {
        "edge_runtime_active": system_audio.transcription_thread is not None and system_audio.transcription_thread.is_alive(),
        "sarvam_connected": bool(os.getenv("SARVAM_API_KEY")),
        "model_loaded": model_loaded,
        "fallback_available": bool(os.getenv("SARVAM_API_KEY")),
        "active_speakers": speakers
    }

@app.websocket("/ws/captions")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            # Poll the result queue from system_audio
            # We do this in a loop inside the websocket handler? 
            # Better to have a background broadcaster, but for simplicity:
            # wait for message? No, we need to push.
            # So we just sleep to keep connection open, and rely on the broadcaster task.
            await asyncio.sleep(1)
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)

async def broadcast_captions():
    """Background task to read queue and broadcast to all clients."""
    while True:
        try:
            # Non-blocking get
            while not system_audio.result_queue.empty():
                res = system_audio.result_queue.get_nowait()
                if isinstance(res, dict):
                    text = res.get("text", "")
                    message = res
                else:
                    text, is_final, latency, speaker = res
                    message = {"text": text, "is_final": is_final, "latency": latency, "speaker": speaker}

                if text:
                    # potentially disconnected clients
                    to_remove = set()
                    for client in list(connected_clients):
                        try:
                            await client.send_json(message)
                        except:
                            to_remove.add(client)
                    
                    for dead in to_remove:
                        try:
                            connected_clients.remove(dead)
                        except KeyError:
                            pass
            
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_captions())

if __name__ == "__main__":
    print("Starting Web Server on port 8080...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
