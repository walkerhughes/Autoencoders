import numpy as np  
from PIL import Image 
import base64
from io import BytesIO
import torch 
import torch.nn as nn
import onnxruntime as ort

version = "0.0.1"

def generate_latent_vectors(num_samples = 1, latent_dim = 64):
    latent_vectors = torch.randn(num_samples, latent_dim)
    return latent_vectors

def generate_vae_30e_image():
    model_path = './decoder_model_30e.onnx'
    ort_sess = ort.InferenceSession(model_path)
    latent_vectors = generate_latent_vectors()
    outputs = ort_sess.run(None, {'input_name': latent_vectors.detach().cpu().numpy()})
    
    # Assuming the output needs to be reshaped and converted to 8-bit unsigned integer
    reshaped = outputs[0][0].transpose(1, 2, 0)
    reshaped = np.clip(reshaped, 0, 1)  # Ensure values are between 0 and 1
    reshaped = (reshaped * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
    
    # Convert numpy array to PIL image
    pil_image = Image.fromarray(reshaped)

    # Save image to a bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode image to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image": img_str}