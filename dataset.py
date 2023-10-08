import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch

def get_dataset(batch_size):
    img_processor = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.MNIST(root='MNIST', train=True, download=True, transform=img_processor, target_transform=target_processor)
    test_set = torchvision.datasets.MNIST(root='MNIST', train=False, download=True, transform=img_processor, target_transform=target_processor)


    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_set = DataLoader(test_set, batch_size=batch_size, num_workers=12)

    return train_set, test_set


def target_processor(label, num_class=10):
    one_hot = torch.zeros((num_class,))
    one_hot[label] = 1
    return one_hot