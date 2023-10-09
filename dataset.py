import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch

registered_dataset = {
    'mnist': torchvision.datasets.MNIST,
    'cifar': torchvision.datasets.CIFAR10,
}

normalization = {
    'mnist': transforms.Normalize((0.1307,), (0.3081,)),
    'cifar': transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
}

def get_dataset(batch_size, dataset):
    train_transform = train_processor(dataset)
    test_transform = test_processer(dataset)
    
    train_set = registered_dataset[dataset](root='./', train=True, download=True, transform=train_transform, target_transform=target_processor)
    test_set = registered_dataset[dataset](root='./', train=False, download=True, transform=test_transform, target_transform=target_processor)
    img_shape = train_set[0][0].shape
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_set = DataLoader(test_set, batch_size=batch_size, num_workers=12)


    return train_set, test_set, img_shape

def train_processor(dataset):
    normalize = normalization[dataset]

    return transforms.Compose([
        # transforms.RandomHorizontalFlip(), this is not desirable for MNIST
        transforms.ToTensor(),
        normalize
    ])

def test_processer(dataset):
    normalize = normalization[dataset]
    return transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


def target_processor(label, num_class=10):
    one_hot = torch.zeros((num_class,))
    one_hot[label] = 1
    return one_hot