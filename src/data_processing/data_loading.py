from torchvision.datasets import FashionMNIST, MNIST
from torchvision import datasets, transforms
from torch.utils import data as tdataset
import random

from src.data_processing import datasets as mydatasets

def get_dataloader(dataset, bs):
    loader = tdataset.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
    )
    return loader

def get_CIFAR10_loader(data_path, config):
    dataset = datasets.CIFAR10(
        root=data_path,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config["bs"])

def get_imagewoof_loader(data_path, config):
    dataset = mydatasets.Imagenette(
        root=data_path,
        csv="noisy_imagewoof.csv",
        transform=transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config.bs)

def get_custom_loader(data_path, config):
    print("running custom dataset")
    dataset = mydatasets.Imagenette(
        root=data_path,
        csv="data.csv",
        transform=transforms.Compose([
            transforms.Resize(config["image_size"]),
            transforms.CenterCrop(config["image_size"]),
            transforms.ToTensor(),
        ])
    )
    return get_dataloader(dataset, config["bs"])

loaders = {
    "CIFAR10": get_CIFAR10_loader,
    "imagewoof": get_imagewoof_loader,
    "custom": get_custom_loader,
}


def get_supported_loader(name):
    return loaders[name]
