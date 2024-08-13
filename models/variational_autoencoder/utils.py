import torch 
import torch.nn as nn 
import os, glob, zipfile

from PIL import Image
from torch.utils.data import Dataset


def unzip_data(path: str = '../../data/cryptopunks/', filename: str = 'cryptopunks.zip'):  
    with zipfile.ZipFile(f'{path}{filename}', 'r') as zip_ref:
        if not os.path.exists(f'{path}imgs'): 
            print("Unzipping images...", end = " ")
            zip_ref.extractall(f'{path}')
            print("done.")
        else: 
            print("Data already unzipped.")


class CryptoPunksDataset(Dataset):
    def __init__(self, root_dir: str = '../../data/cryptopunks/', zip_file_name: str = 'cryptopunks.zip', transform = None):
        unzip_data(path = root_dir, filename = zip_file_name)
        self.img_dir = root_dir
        self.transform = transform
        self.img_paths = glob.glob(os.path.join(root_dir + "imgs", "*.png"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def generate_latent_vectors(num_samples = 1, latent_dim = 64):
    latent_vectors = torch.randn(num_samples, latent_dim)
    return latent_vectors