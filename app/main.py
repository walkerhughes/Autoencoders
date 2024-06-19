from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.model.model import generate_kde_image, generate_vae_image, generate_vae_30e_image, version

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from this origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def home(): 
    return {"status": "OK", "version": version}

@app.post("/predict_kde")
def predict_kde():
    return generate_kde_image()  

@app.post("/predict_vae")
def predict_vae():
    return generate_vae_image()  
    
@app.post("/predict_vae_30e")
def predict_vae():
    return generate_vae_30e_image() 