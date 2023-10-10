import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch

registered_dataset = {
    'mnist': torchvision.datasets.MNIST,
    'cifar': torchvision.datasets.CIFAR10,
}

processor = {
    'mnist_train': transforms.Compose([
        # transforms.RandomHorizontalFlip(), this is not desirable for MNIST
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    'mnist_test': transforms.Compose([
        # transforms.RandomHorizontalFlip(), this is not desirable for MNIST
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]),
    'cifar_train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ]),
    'cifar_test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
}

def get_dataset(batch_size, dataset):
    train_transform = train_processor(f'{dataset}_train')
    test_transform = test_processer(f'{dataset}_test')
    
    train_set = registered_dataset[dataset](root='./', train=True, download=True, transform=train_transform, target_transform=target_processor)
    test_set = registered_dataset[dataset](root='./', train=False, download=True, transform=test_transform, target_transform=target_processor)
    img_shape = train_set[0][0].shape
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_set = DataLoader(test_set, batch_size=batch_size, num_workers=12)


    return train_set, test_set, img_shape

def train_processor(dataset):
    return processor[dataset]

     

def test_processer(dataset):
    return processor[dataset]


def target_processor(label, num_class=10):
    one_hot = torch.zeros((num_class,))
    one_hot[label] = 1
    return one_hot