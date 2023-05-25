import torch
import torchvision.datasets
from torchvision import transforms
from torch.utils.data import DataLoader

from .resnet import MyFlatResNet224, Policy224
from .resnet import MyFlatResNet32, Policy32

from .base import BasicBlock


def cifar10_blockdrop_110(dataPath):
    layer_config = [18, 18, 18]
    agent = resnet.Policy32([1, 1, 1], num_blocks=54)
    model = resnet.MyFlatResNet32(agent, BasicBlock, layer_config, num_classes=10)

    mean_val, std_val = (0, 0, 0), (1, 1, 1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testSet = torchvision.datasets.CIFAR10(
        root=dataPath, train=False, download=True,
        transform=transform_test
    )
    test_loader = DataLoader(testSet, batch_size=1)
    new_dataset = []
    for x, y in test_loader:
        new_dataset.append(x.squeeze(0))
    test_loader = DataLoader(new_dataset, batch_size=1)
    return model, test_loader


def cifar100_blockdrop_110():
    layer_config = [18, 18, 18]
    agent = resnet.Policy32([1, 1, 1], num_blocks=54)
    model = resnet.MyFlatResNet32(agent, base.BasicBlock, layer_config, num_classes=100)

    mean_val, std_val = (0, 0, 0), (1, 1, 1)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataPath = '/disk/CM/Project/Dataset/CIFAR100'
    testSet = torchvision.datasets.CIFAR10(
        root=dataPath, train=False, download=True,
        transform=transform_test
    )
    test_loader = DataLoader(testSet, batch_size=1)
    new_dataset = []
    for x, y in test_loader:
        new_dataset.append(x.squeeze(0))
    test_loader = DataLoader(new_dataset, batch_size=1)
    return model, test_loader


# def imagenet_blockdrop_101():
#     layer_config = [3, 4, 23, 3]
#     rNet = resnet.FlatResNet224(base.Bottleneck, layer_config, num_classes=1000)
#     agent = resnet.Policy224([1, 1, 1, 1], num_blocks=33)
#     return rNet, agent