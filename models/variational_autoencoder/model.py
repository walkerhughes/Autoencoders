import torch
import torch.nn as nn

class VAELoss(nn.Module):
    """
    Binary Cross-Entropy Loss with KL-Divergence regularization. 
    KL-Divergence encourages model to learn latent Standard Normal distribution. 
    """
    def __init__(self):
        super(VAELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, recon_x, x, mean, logvariance):
        bce = self.bce_loss(recon_x, x)
        kl_divergence = -0.5 * torch.sum(1 + logvariance - mean.pow(2) - logvariance.exp())
        loss_ = bce + kl_divergence
        return loss_, bce, kl_divergence

class Encoder(nn.Module):
    """
        Encoder sub-class for Variational Autoencoder (VAE).

        This class takes input images and encodes them into a low-dimensional latent space, 
        represented by mean and log-variance vectors for a multidimensional standard normal distribution. 
        These vectors are used for the reparameterization trick in the VAE's forward pass.

        Attributes:
            mean (nn.Linear): Linear layer to output the mean vector of the latent space.
            logvariance (nn.Linear): Linear layer to output the log-variance vector of the latent space.
            encode (nn.Sequential): Sequential model containing convolutional layers and activation 
                functions to encode the input images.

        Args:
            input_channels (int): Number of channels in the input images.
            hidden_dims (int): Number of hidden dimensions in the convolutional layers.
            latent_dim (int): Dimensionality of the latent space.
    """
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
        """
            Forward pass through the encoder.
            
            Args:
                x (torch.Tensor): Input tensor representing a batch of images.
            
            Returns:
                mean (torch.Tensor): The vector of means of the latent space.
                logvariance (torch.Tensor): The log-variance vector of the latent space.
        """
        x_encoded = self.encode(x)
        x_encoded = x_encoded.view(x_encoded.size(0), -1)
        mean, logvariance = self.mean(x_encoded), self.logvariance(x_encoded)
        return mean, logvariance

class Decoder(nn.Module):
    """
        Decoder sub-class for VAE.
        
        This class takes latent vectors and decodes them back into image space, 
        effectively reversing the encoding process of the Encoder.
        
        Attributes:
            fc (nn.Linear): Linear layer to project the latent space vector into a high-dimensional feature map.
            decode (nn.Sequential): Sequential model containing transposed convolutional layers and activation 
                functions to decode the high-dimensional feature map back into image space.
        
        Args:
            latent_dim (int): Dimensionality of the latent space.
            hidden_dims (int): Number of hidden dimensions in the transposed convolutional layers.
            output_channels (int): Number of channels in the output images.
    """
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
        """
            Forward pass through the decoder.
            
            Args:
                z (torch.Tensor): Input tensor representing the reparameterized 
                    Encoder.mean and Encoder.logvariance vectors.
            
            Returns:
                self.decode(x) (torch.Tensor): The decoded vector. 
        """
        x = self.fc(z)
        x = x.view(x.size(0), -1, 6, 6)  
        return self.decode(x)

class VAE(nn.Module):
    """
        Variational Autoencoder (VAE) Class. 

        This class implements the Encoder and Decoder sub-classes defined above to learn a 
        low-dimensional, latent probability distribution representing the input images it is trained on. 
        Vectors can be sampled from this learned distribution and decoded to produce completely new images. 

        The class implements the reperameterization trick so the model can learn, despite backpropagation 
        being intractable across the latent probability distribution. 

        Args:
            input_channels (int): Number of channels in the input images.
            hidden_dims (int): Number of hidden dimensions in the convolutional layers.
            latent_dim (int): Dimensionality of the latent space.
            output_channels (int): Number of channels in the output images (will typically 
                be the same as input_channels).
    """
    def __init__(self, input_channels, hidden_dims, latent_dim, output_channels):
        super(VAE, self).__init__()
        self.__dict__.update(locals())
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
    
    def sample(self, num_samples, device): 
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decoder(z)