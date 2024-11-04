from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dataclasses import dataclass
from pydantic import BaseModel
from app.model.model import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://whughes.vercel.app"],  # Allow requests from these origins
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

@app.get("/")
def home(): 
    return {"status": "OK", "version": version}
    
@app.post("/predict_10e")
def predict_vae():
    return generate_image_10e()
    
@app.post("/predict_30e")
def predict_vae():
    return generate_image_30e()
    
@app.post("/predict_50e")
def predict_vae():
    return generate_image_50e()


if __name__ == "__main__":
    import os 
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)