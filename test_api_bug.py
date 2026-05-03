import requests
import time
import websocket

print("Connecting to ws")
ws = websocket.WebSocket()
ws.connect("ws://localhost:8080/ws/captions")

print("Starting trans")
res = requests.post("http://localhost:8080/api/start", json={"source_lang":"en", "target_lang":"hi", "model":"base"})
print(res.status_code, res.text)

print("Wait 3s")
time.sleep(3)

print("Changing lang - stops then starts")
res = requests.post("http://localhost:8080/api/stop")
print("STOP", res.status_code, res.text)
res = requests.post("http://localhost:8080/api/start", json={"source_lang":"en", "target_lang":"be", "model":"base"})
print("START", res.status_code, res.text)

print("Refreshing page (reconnecting ws)")
ws.close()
ws = websocket.WebSocket()
ws.connect("ws://localhost:8080/ws/captions")

print("Done")
