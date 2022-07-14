import time
import copy
import torch

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


def train_net(net, dataloaders,
              dataloaders_size,
              device,
              criterion,
              optimizer,
              scheduler,
              num_epochs=25):
    """
    Trains net for epochs given in parameter num_epochs
    meanwhile saving best weights in net.state_dict format
    and returns best parametrized network.
    """
    since = time.time()
    best_net_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    accuracy_stats = {"train": [], "val": []}
    loss_stats = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()
            phase_loss = 0.0
            phase_corrects = 0

            for images, targets in dataloaders[phase]:
                # 1 channel -> 3 channel
                # images = images.repeat(1, 3, 1, 1).to(device)
                images = images.to(device)
                labels = targets["labels"].to(device)
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(images)
                    if str(criterion) == 'CrossEntropyLoss()':
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if str(criterion) == 'BCELoss()':
                        preds = torch.sigmoid(outputs).reshape(-1).detach(
                            ).cpu().numpy().round()
                        loss = criterion(torch.sigmoid(outputs).reshape(-1),
                                         labels)

                    # backward pass + opt
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                phase_loss += loss.item() * images.size(0)
                if str(criterion) == 'BCELoss()':
                    # print(preds, labels)
                    phase_corrects += torch.sum(torch.tensor(preds)
                                                == labels.cpu())
                if str(criterion) == 'CrossEntropyLoss()':
                    phase_corrects += torch.sum(preds == labels)

            if phase == 'train':
                scheduler.step()

            epoch_loss = phase_loss / dataloaders_size[phase]
            epoch_acc = phase_corrects.double() / dataloaders_size[phase]
            loss_stats[phase].append(epoch_loss)
            accuracy_stats[phase].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the net params
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_net_wts = copy.deepcopy(net.state_dict())

        print()

    time_e = time.time() - since
    print(f'Training completed in {time_e // 60:.0f}m {time_e % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best net weights
    net.load_state_dict(best_net_wts)
    return net, loss_stats, accuracy_stats


def test_net(net, data_loaders: dict, class_names: list, device):

    net.eval()
    preds = np.array([])
    true = np.array([])
    l1 = list()
    l2 = list()
    l3 = list()
    l4 = list()

    for batches in data_loaders["test"]:

        images = batches[0].to(device)
        # images = images.repeat(1, 3, 1, 1)
        targets = batches[1]['labels'].to(device)

        with torch.no_grad():
            outputs = net(images)
            if outputs.size(dim=1) == 1:
                preds = np.append(preds, torch.sigmoid(
                    outputs).reshape(-1).detach().cpu().numpy().round())
            elif outputs.size(dim=1) == 4:
                preds = np.append(preds, outputs.softmax(1).argmax(1).cpu())
            true = np.append(true, targets.cpu())

        for img in range(len(targets)):
            p = torch.sigmoid(
                    outputs).reshape(-1).detach().cpu().numpy().round()
            if batches[1]["class"][img] == 'Normal':
                l1.append(p[img])
            if batches[1]["class"][img] == 'Benign':
                l2.append(p[img])
            if batches[1]["class"][img] == 'Malignant':
                l3.append(p[img])
            if batches[1]["class"][img] == 'Lymph_nodes':
                l4.append(p[img])

    print(f'l1:{l1}\nSum:{sum(i for i in l1)}\n'
          + f'l2:{l2}\nSum:{sum(i for i in l2)}\n'
          + f'l3:{l3}\nSum:{sum(i for i in l3)}\n'
          + f'l4:{l4}\nSum:{sum(i for i in l4)}')
    cm = confusion_matrix(true, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    print(classification_report(true, preds, target_names=class_names))
    return
