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

@dataclass 
class DecoderModelPath(BaseModel): 
    path: str 

@dataclass 
class ImageStr(BaseModel): 
    image_string: str 

@app.get("/")
def home(): 
    return {"status": "OK", "version": version}
    
@app.post("/predict", response_model = ImageStr)
def predict_vae(model_path: DecoderModelPath):
    return generate_vae_image(model_path.path)