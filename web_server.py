from fastapi import FastAPI, WebSocket, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
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
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "languages": LANGUAGE_NAMES,
        "v": int(time.time())
    })

@app.post("/api/start")
async def start_transcription(data: dict):
    print(f"[DEBUG] /api/start payload: {data}")
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "hi")
    model = data.get("model", "base.en")
    
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
                result = system_audio.result_queue.get_nowait()
                text = result[0]
                is_final = result[1]
                src_id = result[2] if len(result) > 2 else "Local"
                if text:
                    message = {"text": text, "is_final": is_final, "source": src_id}
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
