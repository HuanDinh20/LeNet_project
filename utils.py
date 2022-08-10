import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch import nn
from torch.utils.data import DataLoader


def xavier_transform(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)


def image_augmentation():
    """Only Horizontal and ToTensor"""
    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    return aug


def create_dataloader(train_set, test_set, batchsize):
    train_dataloader = DataLoader(train_set, batchsize)
    test_dataloader = DataLoader(test_set, batchsize)
    return train_dataloader, test_dataloader


def get_FashionMNist(augmentation, download=True):
    train_set = FashionMNIST(root='./data', train=True, transform=augmentation,
                             download=download)
    val_set = FashionMNIST(root='./data', train=False, transform=transforms.ToTensor(),
                           download=download)
    return train_set, val_set


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def load_model(model, path):
    loaded_model = torch.load(path)
    return loaded_model


def model_predict_value(model, test_set, classes, range_of_predictions):
    for i in range(range_of_predictions):
        x, y = test_set[i][0], test_set[i][1]
        x = torch.unsqueeze(x, 0).to(torch.device('cuda'))
        with torch.no_grad():
            pred = model(x)
            predicted, actual = classes[pred[0].argmax(0)], classes[y]
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
