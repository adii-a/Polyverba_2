from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello"}

if __name__ == "__main__":
    print("Starting uvicorn...")
    uvicorn.run(app, port=8001)
