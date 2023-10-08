import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torch

registered_dataset = {
    'mnist': torchvision.datasets.MNIST,
    'cifar': torchvision.datasets.CIFAR10,
}

def get_dataset(batch_size, dataset):
    train_set = registered_dataset[dataset](root='./', train=True, download=True, transform=img_processor, target_transform=target_processor)
    test_set = registered_dataset[dataset](root='./', train=False, download=True, transform=img_processor, target_transform=target_processor)
    img_shape = train_set[0][0].shape

    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_set = DataLoader(test_set, batch_size=batch_size, num_workers=12)


    return train_set, test_set, img_shape

def img_processor(img):
    return transforms.Compose([
        transforms.ToTensor(),
    ])(img)



def target_processor(label, num_class=10):
    one_hot = torch.zeros((num_class,))
    one_hot[label] = 1
    return one_hot