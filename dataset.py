import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms

def get_dataset(batch_size):
    processor = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root='MNIST', train=True, download=True, transform=processor)
    test_set = torchvision.datasets.MNIST(root='MNIST', train=False, download=True, transform=processor)


    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_set = DataLoader(test_set, batch_size=batch_size, num_workers=4)

    return train_set, test_set
