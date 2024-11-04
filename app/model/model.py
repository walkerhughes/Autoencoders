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

def generate_image_10e(): 
    ort_sess = ort.InferenceSession('./decoder_model_10e.onnx')
    latent_vectors = generate_latent_vectors()
    outputs = ort_sess.run(None, {'input_name': latent_vectors.detach().cpu().numpy()})
    
    reshaped = outputs[0][0].transpose(1, 2, 0)
    reshaped = np.clip(reshaped, 0, 1)  
    reshaped = (reshaped * 255).astype(np.uint8)  
    
    pil_image = Image.fromarray(reshaped)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": img_str}

def generate_image_30e(): 
    ort_sess = ort.InferenceSession('./decoder_model_30e.onnx')
    latent_vectors = generate_latent_vectors()
    outputs = ort_sess.run(None, {'input_name': latent_vectors.detach().cpu().numpy()})
    
    reshaped = outputs[0][0].transpose(1, 2, 0)
    reshaped = np.clip(reshaped, 0, 1)  
    reshaped = (reshaped * 255).astype(np.uint8)  
    
    pil_image = Image.fromarray(reshaped)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": img_str}

def generate_image_50e(): 
    ort_sess = ort.InferenceSession('./decoder_model_50e.onnx')
    latent_vectors = generate_latent_vectors()
    outputs = ort_sess.run(None, {'input_name': latent_vectors.detach().cpu().numpy()})
    
    reshaped = outputs[0][0].transpose(1, 2, 0)
    reshaped = np.clip(reshaped, 0, 1)  
    reshaped = (reshaped * 255).astype(np.uint8)  
    
    pil_image = Image.fromarray(reshaped)

    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    return {"image": img_str}