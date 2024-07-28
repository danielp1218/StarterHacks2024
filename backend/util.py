import torch.nn as nn
import torch
import glob
import os
import torch.optim as optim
import torch.optim.optimizer
import pathlib
import numpy as np
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image


"""
json = {
    'layers': [
        {'name': string, 'in_channels': int, 'out_channels': int, 'kernel_size': [int, int]},
    ],
    'optimizer': {
        'type': string,
        'lr': float,
    },
    'loss': {
        'type': string,
    },
    'reduceLrOnPlateau': {
        'type': string,
    }

}
"""



stringToLayer = {
    'flatten': nn.Flatten,
    'conv2d': nn.Conv2d,
    'linear': nn.Linear,
    'softmax': nn.Softmax,
    'relu': nn.ReLU,
    'maxpool2d': nn.MaxPool2d
}

layerTypes = {
    'linear': ['linear'],
    'convolution': ['conv2d'],
    'activation': ['relu', 'softmax', 'flatten'],
    'pooling': ['maxpool2d']
}

def json_to_layers(json: dict) -> list[nn.Module]:
    layers = json['layers']
    model = []
    for layer in layers:
        lType = layer['type']
        if lType in layerTypes['linear']:
            model.append(stringToLayer[lType](layer['in_channels'], layer['out_channels']))
        elif lType in layerTypes['convolution']:
            model.append(stringToLayer[lType](layer['in_channels'], layer['out_channels'], *layer['kernel_size']))
        elif lType in layerTypes['activation']:
            model.append(stringToLayer[lType]())
        elif lType in layerTypes['pooling']:
            model.append(stringToLayer[lType](*layer['kernel_size']))
        else:
            print(lType," layer type not supported")
    return model


def json_to_optimizer(json: dict, parameters) -> torch.optim.Optimizer:
    optimizer = json['optimizer']
    if optimizer['type'] == 'sgd':
        return optim.SGD(params=parameters, lr=optimizer['lr'])
    elif optimizer['type'] == 'adam':
        return optim.Adam(params=parameters, lr=optimizer['lr'], betas=[0.9, 0.99])
    elif optimizer['type'] == 'adagrad':
        return optim.Adagrad(params=parameters, lr=optimizer['lr'])
    else:
        raise ValueError(f'Optimizer {optimizer["type"]} not supported')
    
def json_to_criterion(json: dict) -> nn.Module:
    loss = json['loss']
    if loss['type'] == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif loss['type'] == 'MSELoss':
        return nn.MSELoss()
    else:
        raise ValueError(f'Loss {loss["type"]} not supported')


class Model(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.stack(x)
        return logits

class CustomDataset(Dataset):
    def __init__(self, root_dir, image_dimensions=(28, 28)):
        self.image_size = image_dimensions
        self.image_paths = []
        self.labels = []
        self.resize = transforms.Resize(self.image_size)
        for dir in os.listdir(root_dir):
            self.labels.append(dir)
            for file in os.listdir(os.path.join(root_dir, dir)):
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.image_paths.append(os.path.join(root_dir, dir, file))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #img = read_image(self.image_paths[idx], ImageReadMode.RGB).float()
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # print(self.image_size)
        img = self.resize(img)
        arr = np.array(img).astype('float32')
        cls = os.path.basename(os.path.dirname(self.image_paths[idx]))
        idx = self.labels.index(cls)

        return torch.tensor(arr), torch.tensor(idx)


def getDataLoaders(dataset: CustomDataset):
    splits = [0.7, 0.2, 0.1]
    split_sizes = []
    for sp in splits[:-1]:
        split_sizes.append(int(sp * len(dataset)))
    split_sizes.append(len(dataset) - sum(split_sizes))
    train_set, test_set, val_set = torch.utils.data.random_split(dataset, split_sizes)

    return {
        "train": DataLoader(train_set, batch_size=16, shuffle=True),
        "test": DataLoader(test_set, batch_size=16, shuffle=False),
        "val": DataLoader(val_set, batch_size=16, shuffle=False)
    }

def train(criterion, opt, model, trainloader, epochs=2, device='cpu'):
    running_loss = 0.0

    for e in range(epochs):
        print(f'Epoch {e + 1}')
        count = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            count += 1
            if count % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{e + 1}, {count + 1:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0


testJson = {
    'layers': [
        {'type': 'flatten'},
        {'type': 'conv2d', 'in_channels': 28*28*3, 'out_channels': 256, 'kernel_size': [3, 3]},
        {'type': 'relu'},
        {'type': 'linear', 'in_channels': 253, 'out_channels': 253},
        {'type': 'relu'},
        {'type': 'linear', 'in_channels': 253, 'out_channels': 253},
        {'type': 'relu'},
        {'type': 'linear', 'in_channels': 253, 'out_channels': 2}
    ],
    'optimizer': {
        'type': 'sgd',
        'lr': 0.001,
    },
    'loss': {
        'type': 'CrossEntropyLoss'
    },
    'reduceLrOnPlateau': {
        'type': 'ReduceLROnPlateau',
    }
}

def test():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = torch.load('model.pt')
    dataset = CustomDataset('data/test')
    loaders = getDataLoaders(dataset)
    testloader = loaders['train']
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} test images: {100 * correct / total}%')


def run(json):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    device = torch.device(device)

    #mnist = [(np.array(data[0]).astype('float32'),data[1]) for data in datasets.MNIST(root='data', train=True, download=True)]
    #print(len(mnist))
    #trainloader = DataLoader(mnist, batch_size=4, shuffle=True)
    model = Model(json_to_layers(json)).to(device)
    opt = json_to_optimizer(json, model.parameters())
    criterion = json_to_criterion(json)

    
    dataset = CustomDataset('data/train')
    loaders = getDataLoaders(dataset)
    trainloader = loaders['train']
    train(criterion, opt, model, trainloader, epochs=2, device=device)

    torch.save(model, 'model.pt')