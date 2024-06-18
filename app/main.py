from fastapi import FastAPI
from app.model.model import generate_image, version

app = FastAPI() 

@app.get("/")
def home(): 
    return {"status": "OK", "version": version}

@app.post("/predict")
def predict_endpoint():
    result = generate_image()  
    return result