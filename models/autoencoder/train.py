import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision.transforms as transforms 

from autoencoder import Autoencoder
from utils import get_mnist_train_loader

import tqdm 
import argparse 

if __name__ == "__main__": 

    # example usage: python models/autoencoder/train.py 
    parser = argparse.ArgumentParser(description = "Train Autoencoder on CryptoPunks")
    parser.add_argument('-input_dim', type = int, default = 784, help = 'dimension of flattened image')
    parser.add_argument('-hidden_dim', type = int, default = 128, help = 'dimension of hidden layer in autoencoder network')
    parser.add_argument('-latent_dim', type = int, default = 64, help = 'dimension of latend layer in autoencoder network')
    parser.add_argument('-learning_rate', type = float, default = 1e-3, help = 'learning rate for Adam optimizer')
    parser.add_argument('-batch_size', type = float, default = 64, help = 'batch size during training')
    parser.add_argument('-epochs', type = int, default = 20, help = 'Number of epochs to train')
    args = parser.parse_args()

    print("getting training data...", end = " ")
    train_loader = get_mnist_train_loader(
        batch_size=args.batch_size
    ) 
    print("training data loaded.")

    model = Autoencoder(
        input_dim = args.input_dim, 
        hidden_dim = args.hidden_dim, 
        latent_dim = args.latent_dim
    )
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(
        params = model.parameters(), 
        lr = args.learning_rate
    )

    losses = [] 
    print("starting training...")
    for epoch in range(args.epochs): 
        # for data in train_loader: 
        epoch_loss = [] 
        for i, data in (progress := tqdm.tqdm(enumerate(train_loader))): 
            image, _ = data 
            # flatten image to a vector 
            image = image.view(image.size(0), -1)
            # forward pass + calculate loss 
            output = model(image) 
            loss = criterion(output, image)
            epoch_loss.append(loss.item())

            # backward pass 
            optimizer.zero_grad() 
            loss.backward()  
            optimizer.step() 

            # update progress par for each image 
            progress.set_description(f"Epoch: {epoch + 1}, Image: {i + 1}/{args.batch_size}, Loss: {loss.item()}")
        losses.append(epoch_loss)

    print("training complete.")

    # Save the trained model
    torch.save(model.state_dict(), './models/autoencoder/autoencoder.pth')
