import joblib 
import numpy as np  
from PIL import Image 
import base64
from io import BytesIO
import torch 
import torch.nn as nn

version = "0.0.1"

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dim):
        super(Encoder, self).__init__()
        self.mean = nn.Linear(hidden_dims * 2 * 6 * 6, latent_dim)  
        self.logvariance = nn.Linear(hidden_dims * 2 * 6 * 6, latent_dim)  
        self.encode = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dims, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dims, hidden_dims * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x_encoded = self.encode(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        mean, logvariance = self.mean(x_encoded), self.logvariance(x_encoded)
        return mean, logvariance


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dims * 2 * 6 * 6)  
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims * 2, hidden_dims, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.ConvTranspose2d(hidden_dims, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), -1, 6, 6)  
        return self.decode(x)

class VAE(nn.Module):
    def __init__(self, input_channels, hidden_dims, latent_dim, output_channels):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, output_channels)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar
    
    
def generate_kde_image():

    kde = joblib.load("./kde_model_compressed.joblib")

    new_image = kde.sample()
    reshaped = new_image.reshape(24, 24, 4)/255
    reshaped = np.clip(reshaped, 0, 1)*255
    reshaped = reshaped.astype(np.uint8)

    # Convert numpy array to PIL image
    pil_image = Image.fromarray(reshaped)

    # Save image to a bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode image to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image": img_str}


def sample(model, num_samples, device):
    z = torch.randn(num_samples, model.encoder.mean.out_features).to(device)
    return model.decoder(z)


def generate_vae_30e_image():

    model = VAE(input_channels=4, hidden_dims=128, latent_dim=64, output_channels=4)

    model.load_state_dict(
        torch.load('./variational_autoencoder_30_epochs.pth', map_location=torch.device('cpu'))
    ) 

    samples = sample(model, num_samples=1, device="cpu")
    samples = samples.detach().cpu().numpy()
    new_image = samples[0].transpose(1, 2, 0).astype(float)
    reshaped = np.clip(new_image, 0, 1)*255
    reshaped = reshaped.astype(np.uint8)

    # Convert numpy array to PIL image
    pil_image = Image.fromarray(reshaped)

    # Save image to a bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode image to base64
    img_str = base64.b64encode(buffer.read()).decode('utf-8')

    return {"image": img_str}