import joblib 
import numpy as np  
from PIL import Image 
import base64
from io import BytesIO

version = "0.0.1"

def generate_image():

    kde = joblib.load("./kde_model.joblib")

    new_image = kde.sample()
    reshaped = new_image.reshape(24, 24, 4).astype(np.float64)/255

    # Convert numpy array to PIL image
    pil_image = Image.fromarray(reshaped)

    # Save image to a bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode image to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image": img_str}