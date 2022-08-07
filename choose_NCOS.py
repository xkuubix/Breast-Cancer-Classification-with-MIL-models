import torch
from torch import nn, optim
from torchvision import models
from simple_MIL import SimpleMIL
from gated_MIL import GatedMIL
from torch.optim import lr_scheduler


def choose_NCOS(net_ar: str, device,
                pretrained,
                criterion_type: str,
                optimizer_type: str,
                lr, wd, scheduler):

    '''
    net_ar: resnet18, resnet50, alexnet, vgg16
    criterion_type: bce, ce
    optimizer_type: sgd, adam
    scheduler_type: lin, [to be added]
    '''

    # SELECT CRITERION
    if criterion_type == 'bce':
        criterion = nn.BCELoss()
        num_out = 1
    elif criterion_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        num_out = 4  # lub 3
    criterion.to(device)

    # SELECT NET ARCHITECTURE
    if net_ar == 'resnet18':
        net = models.resnet18(pretrained=pretrained)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'resnet50':
        net = models.resnet50(pretrained=pretrained)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'alexnet':
        net = models.alexnet(pretrained=pretrained)
        net.classifier[4] = nn.Linear(4096, 1024)
        net.classifier[6] = nn.Linear(1024, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)
        # to bc

    elif net_ar == 'vgg16':
        net = models.vgg16(pretrained=pretrained)
        # to bc

    elif net_ar == 'mil':
        net = SimpleMIL(num_classes=num_out,
                        num_instances=50, pretrained=pretrained)
        # if torch.cuda.device_count() == 2:
        # net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'gmil':
        net = GatedMIL(num_classes=num_out,
                       pretrained=pretrained)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    # SELECT OPTIMIZER

    if optimizer_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # SELECT SCHEDULER
    if scheduler['name'] == 'lin':
        step_szie = scheduler['step_size']
        gamma = scheduler['gamma']
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=step_szie,
                                        gamma=gamma)
    elif scheduler['name'] == 'one-cycle':
        STEPS_PER_EPOCH = scheduler['steps_per_epoch']
        TOTAL_STEPS = scheduler['epochs'] * STEPS_PER_EPOCH
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                            total_steps=TOTAL_STEPS)

    return net, criterion, optimizer, scheduler