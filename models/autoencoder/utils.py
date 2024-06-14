import torch 
import torch.utils
import torchvision 
import torchvision.transforms as transforms 

def get_mnist_train_loader(train: bool = True, shuffle: bool = True, batch_size: int = 64): 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=train,
        transform=transform,
        download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    return train_loader