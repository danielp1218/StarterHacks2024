import torch.nn as nn
import torch

import glob
import os
import torch.optim as optim
from torch.optim import sgd
import torch.optim.optimizer
import pathlib

import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

# https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
"""
json = {
    'layers': [
        {'name': string, 'in_channels': int, 'out_channels': int, 'kernel_size': [int, int]},
    ],
    'optimizer': {
        'type': string,
        'lr': float,
        'momentum': float
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
        
        if layer in layerTypes['linear']:
            model.append(stringToLayer[layer['type']](layer['in_channels'], layer['out_channels']))
        elif layer in layerTypes['convolution']:
            model.append(stringToLayer[layer['type']](layer['in_channels'], layer['out_channels'], *layer['kernel_size']))
        elif layer in layerTypes['activation']:
            model.append(stringToLayer[layer['type']]())
        elif layer in layerTypes['pooling']:
            model.append(stringToLayer[layer['type']](*layer['kernel_size']))
    return model


def json_to_optimizer(json: dict, model: nn.Module) -> torch.optim.optimizer.Optimizer:
    optimizer = json['optimizer']
    if optimizer['type'] == 'SGD':
        return optim.sgd.SGD(model.parameters(), lr=optimizer['lr'], momentum=optimizer['momentum'])
    elif optimizer['type'] == 'Adam':
        return optim.adam.Adam(model.parameters(), lr=optimizer['lr'])
    elif optimizer['type'] == 'Adagrad':
        return optim.adagrad.Adagrad(model.parameters(), lr=optimizer['lr'])
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

# I would like to direct your attention to one thing. In line 7, you may notice that the final linear layer has 3 output neurons. This is because the example I mentioned in the beginning has 3 classes (cat/dog/rabbit).
# auto do that
class Net(nn.Module):
    def __init__(self, layers: list[nn.Module], num_of_categories: int):
        super().__init__()
        # how to integrate num of categories into the last layer ???
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.stack(x)
        return logits

class CustomDataset(Dataset):
    def __init__(self, root_dir):
        # self.transform = transform
        self.image_paths = []
        for ext in ['png', 'jpg']:
            # this needs testing !!
            self.image_paths += str(pathlib.Path(__file__).parent / root_dir / '*' / f'*.{ext}')
            # glob.glob(os.path.join(root_dir, '*', f'*.{ext}'))
        class_set = set()
        for path in self.image_paths:
            class_set.add(os.path.dirname(path))
        self.class_lbl = { cls: i for i, cls in enumerate(sorted(list(class_set)))}
        print(self.class_lbl)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx], ImageReadMode.RGB).float()
        cls = os.path.basename(os.path.dirname(self.image_paths[idx]))
        label = self.class_lbl[cls]

        return img, torch.tensor(label)
        # return self.transform(img), torch.tensor(label)


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


# loss = {

# }

def train(criterion, optimizer, root_dir, trainloader):
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # type: ignore

    dataset = CustomDataset(root_dir)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


testJson = {
    'layers': [
        {'type': 'flatten'},
        {'type': 'linear', 'in_channels': 28*28, 'out_channels': 512},
        {'type': 'relu'},
        {'type': 'linear', 'in_channels': 120, 'out_channels': 84},
        {'type': 'linear', 'in_channels': 84, 'out_channels': 10}
    ],
    'optimizer': {
        'type': 'SGD',
        'lr': 0.001,
    },
    'loss': {
        'type': 'CrossEntropyLoss'
    },
}

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def run():
    net = Net(json_to_layers(testJson), 2).to(device)
    opt = json_to_optimizer(testJson, net)
    criterion = json_to_criterion(testJson)

    trainloader = getDataLoaders(CustomDataset('data'))['train']
    train(criterion, opt, 'data', trainloader)
    torch.save(net, 'model.pt')