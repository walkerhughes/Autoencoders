# Variational Autoencoders + Kernel Density Estimation + CryptoPunks + Model Deployment

# Project Features
> 1. Variational Autoencoder class written in PyTorch trained on simple images 
>> a. Custom loss function inheriting from `torch.nn.BCELoss` with KL-divergence acting as regularization term to learn Standard Normal distribution
> 2. Trained Kernel Density Estimator on simple images
> 3. Both models deployed with FastAPI and Heroku Cloud Service on `whughes.vercel.app`
