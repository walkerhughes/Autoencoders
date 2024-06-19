import torch
import torch.nn as nn


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
    
    