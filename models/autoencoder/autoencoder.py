import torch.nn as nn 

class Autoencoder(nn.Module): 
    """
    Simple Autoencoder class for images. 

    Inputs
        input_dim (int): dimension of flattened image 
        hidden_dim (int): dimension of hidden layer in autoencoder network 
        latent_dim (int): dimension of latend layer in autoencoder network 

        encoder: input_dim -> hidden_dim -> latent_dim
        deencoder: latent_dim -> hidden_dim -> input_dim
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int): 
        super(Autoencoder, self).__init__() 
        self.__dict__.update(locals())
        # encoder framework 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        # decoder framework 
        self.dencoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x): 
        # perform encoder and decoder steps on image x 
        encoding = self.encoder(x)
        decoding = self.dencoder(encoding)
        return decoding 
