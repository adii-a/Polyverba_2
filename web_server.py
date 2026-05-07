from fastapi import FastAPI, WebSocket, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import uvicorn
import asyncio
import queue
import time
import threading
import stt.system_audio as system_audio
from stt.translate import FLORES_CODES, LANGUAGE_NAMES

# Default model to preload — matches the model used in /api/start
PRELOAD_MODEL = "base"

async def _notify_model_ready():
    """Wait for preload thread to finish, then push a system event to all WS clients."""
    loop = asyncio.get_event_loop()
    # Poll until model_ready or timeout (120s)
    for _ in range(240):
        if system_audio.model_ready:
            break
        await asyncio.sleep(0.5)

    msg = {"type": "system", "event": "model_ready", "ready": system_audio.model_ready}
    for client in list(connected_clients):
        try:
            await client.send_json(msg)
        except Exception:
            pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Whisper preload in a background thread immediately
    threading.Thread(
        target=system_audio.preload_whisper,
        args=(PRELOAD_MODEL,),
        daemon=True,
        name="whisper-preload"
    ).start()

    # Start caption broadcaster + readiness notifier
    asyncio.create_task(broadcast_captions())
    asyncio.create_task(_notify_model_ready())
    yield

app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

templates = Jinja2Templates(directory="web/templates")

# Store connected websockets
connected_clients = set()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "languages": LANGUAGE_NAMES,
        "v": int(time.time())
    })

@app.get("/api/status")
async def get_status():
    return {"model_ready": system_audio.model_ready}

@app.post("/api/start")
async def start_transcription(data: dict):
    # Block start if model hasn't finished loading yet
    if not system_audio.model_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "message": "Model is still loading, please wait a moment..."}
        )

    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "hi")
    model = data.get("model", "base")

    # Auto-translate logic
    translate = (source_lang != target_lang)

    success = system_audio.start_transcription(source_lang, target_lang, model, translate)
    if success:
        return {"status": "started", "message": f"Started {source_lang}->{target_lang}"}
    else:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Failed to start"})

@app.post("/api/stop")
async def stop_transcription():
    system_audio.stop_transcription()
    return {"status": "stopped"}

@app.websocket("/ws/captions")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    # Immediately tell this client whether the model is ready
    await websocket.send_json({
        "type": "system",
        "event": "model_ready",
        "ready": system_audio.model_ready
    })
    try:
        while True:
            await asyncio.sleep(1)
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)

async def broadcast_captions():
    """Background task to read queue and broadcast to all clients."""
    while True:
        try:
            while not system_audio.result_queue.empty():
                result = system_audio.result_queue.get_nowait()
                text = result[0]
                is_final = result[1]
                src_id = result[2] if len(result) > 2 else "Local"
                latency = result[3] if len(result) > 3 else 0.0
                if text:
                    message = {"text": text, "is_final": is_final, "source": src_id, "latency": latency}
                    to_remove = set()
                    for client in list(connected_clients):
                        try:
                            await client.send_json(message)
                        except Exception:
                            to_remove.add(client)
                    for dead in to_remove:
                        connected_clients.discard(dead)

            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"Broadcast error: {e}")
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    print("Starting Web Server on port 8080...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8080)
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()
