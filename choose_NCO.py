import torch
from torch import nn, optim
from torchvision import models
from simple_MIL import SimpleMIL
from gated_MIL import GatedMIL


def choose_NCO(net_ar: str, device,
               pretrained,
               criterion_type: str,
               optimizer_type: str,
               lr=1e-3, wd=1e-1, params=None):

    '''
    net_ar: resnet18, resnet50, alexnet, vgg16
    criterion_type: bce, ce
    optimizer_type: sgd, adam
    '''

    if criterion_type == 'bce':
        criterion = nn.BCELoss()
        num_out = 1
    elif criterion_type == 'ce':
        criterion == nn.CrossEntropyLoss()
        num_out = 4  # lub 3

    if net_ar == 'resnet18':
        net = models.resnet18(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net = net.to(device)
    elif net_ar == 'resnet50':
        net = models.resnet50(pretrained=True)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net = net.to(device)
    elif net_ar == 'alexnet':
        net = models.alexnet(pretrained=True)
        net.classifier[4] = nn.Linear(4096, 1024)
        net.classifier[6] = nn.Linear(1024, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net = net.to(device)
        # to bc
    elif net_ar == 'vgg16':
        net = models.vgg16(pretrained=True)
        # to bc
    elif net_ar == 'mil':
        net = SimpleMIL(num_classes=num_out,
                        num_instances=50, pretrained=True)
        if 0:  # torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net = net.to(device)
    elif net_ar == 'gmil':
        net = GatedMIL(num_classes=num_out,
                       num_instances=75, pretrained=True)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net = net.to(device)

    if criterion_type == 'bce':
        criterion = nn.BCELoss()
    elif criterion_type == 'ce':
        criterion == nn.CrossEntropyLoss()

    if params is None:
        params = net.parameters()

    if optimizer_type == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=wd)

    return net, criterion, optimizer
