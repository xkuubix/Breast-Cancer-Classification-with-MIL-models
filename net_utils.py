import time
import copy
import torch

import numpy as np
import matplotlib.pyplot as plt
# from deactivate_batchnorm import deactivate_batchnorm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc


def train_net(net, dataloaders,
              dataloaders_size,
              device,
              criterion,
              optimizer,
              scheduler,
              num_epochs,
              neptune_run):
    """
    Trains net for epochs given in parameter num_epochs
    meanwhile saving best weights in net.state_dict format
    and returns best parametrized network.
    """
    since = time.time()
    best_net_wts = copy.deepcopy(net.state_dict())
    best_loss = None
    best_acc = None
    early_stopping_counter = 0
    patience = 75
    accuracy_stats = {"train": [], "val": []}
    loss_stats = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1 :3}/{num_epochs}', end=' ')

        pe_ar_list = ['APE_SAMIL', 'DSMIL', 'GatedMIL']

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                # net.apply(deactivate_batchnorm)
            else:
                net.eval()
            phase_loss = 0.0
            phase_corrects = 0

            for images, targets in dataloaders[phase]:
                images = images.to(device)
                labels = targets["labels"].to(device)
                optimizer.zero_grad()
                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    if net.__class__.__name__ in pe_ar_list:
                        outputs = net(images, targets["tile_cords"])
                    else:
                        outputs = net(images)
                    if str(criterion) == 'CrossEntropyLoss()':
                        _, preds = torch.max(outputs[0].reshape(-1, 4), 1)

                        if type(net).__name__ == 'DSMIL':
                            max_prediction, _ = torch.max(outputs[3], 1)
                            loss_bag = criterion(outputs[0].reshape(-1, 4),
                                                 labels)

                            loss_max = criterion(max_prediction.reshape(-1, 4),
                                                 labels)

                            loss_total = 0.5*loss_bag + 0.5*loss_max
                            loss = loss_total.mean()
                        else:
                            loss = criterion(outputs[0].reshape(-1, 4), labels)

                    if str(criterion) == 'BCELoss()':
                        # preds = torch.sigmoid(outputs).reshape(-1).detach(
                        #     ).cpu().numpy().round()
                        preds = torch.sigmoid(outputs[0]).reshape(-1).detach(
                              ).cpu().numpy().round()

                        if type(net).__name__ == 'DSMIL':
                            max_prediction, _ = torch.max(outputs[3], 1)

                            loss_bag = criterion(
                                torch.sigmoid(outputs[0]),
                                labels.view(-1, 1))

                            loss_max = criterion(
                                torch.sigmoid(max_prediction),
                                labels.view(-1, 1))

                            loss_total = 0.5*loss_bag + 0.5*loss_max
                            loss = loss_total.mean()
                        else:
                            loss = criterion(torch.sigmoid(
                                outputs[0]).reshape(-1), labels)

                    # backward pass + opt
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()  # ----------------------------

                # statistics
                phase_loss += loss.item() * images.size(0)
                if str(criterion) == 'BCELoss()':
                    # print(preds, labels)
                    phase_corrects += torch.sum(torch.tensor(preds)
                                                == labels.cpu())
                if str(criterion) == 'CrossEntropyLoss()':
                    phase_corrects += torch.sum(preds == labels)

            if phase == 'train':
                LR_val = scheduler.get_last_lr()[0]
                scheduler.step()  # ----------------------------

            epoch_loss = phase_loss / dataloaders_size[phase]
            epoch_acc = phase_corrects.double() / dataloaders_size[phase]

            if neptune_run is not None:
                # NEPTUNE LOGGING
                # print(phase + '/loss')
                neptune_run[phase + '/loss'].log(epoch_loss)
                # print(neptune_run[phase + '/loss'])
                neptune_run[phase + '/accuracy'].log(epoch_acc)

            loss_stats[phase].append(epoch_loss)
            accuracy_stats[phase].append(epoch_acc)

            time_e = time.time() - since
            print(f'| {phase} loss: {epoch_loss:.4f} - ' +
                  f'acc: {epoch_acc:.4f}', end=' ')
            if phase == 'val':
                print(f'| LR: {LR_val:.6f} | '
                      + f'time: {time_e // 60:3.0f}m {time_e % 60:2.0f}s')

            # deep copy the net params
            if phase == 'val' and best_loss is None:
                best_loss = epoch_loss
                best_net_wts = copy.deepcopy(net.state_dict())
                early_stopping_counter = 0
            elif phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_net_wts = copy.deepcopy(net.state_dict())
                early_stopping_counter = 0
            elif phase == 'val' and epoch_loss > best_loss:
                early_stopping_counter += 1

            if phase == 'val' and epoch_acc > best_acc or best_acc is None:
                best_acc = epoch_acc
                best_net_wts_acc = copy.deepcopy(net.state_dict())

        # early stopping
        if early_stopping_counter >= patience:
            print('INFO: Early stopping!')
            break

    time_e = time.time() - since
    print(f'Training completed in {time_e // 60:.0f}m {time_e % 60:.0f}s')
    print(f'Best val Loss: {best_loss:4f}')

    # load best net weights
    # net.load_state_dict(best_net_wts)

    return best_net_wts, best_net_wts_acc, loss_stats, accuracy_stats


def test_net(net, data_loaders: dict, class_names: list, device):

    net.eval()
    scores = np.array([])
    preds = np.array([])
    true = np.array([])

    pe_ar_list = ['APE_SAMIL', 'DSMIL', 'GatedMIL']

    for images, targets in data_loaders["test"]:

        images = images.to(device)
        labels = targets['labels'].to(device)

        with torch.no_grad():
            if net.__class__.__name__ in pe_ar_list:
                outputs = net(images, targets["tile_cords"])
            else:
                outputs = net(images)
            if len(outputs) > 1:
                outputs = outputs[0]
            if outputs.size(dim=1) == 1:
                score = torch.sigmoid(
                    outputs).reshape(-1).detach().cpu().numpy()
                scores = np.append(scores, score)
            elif outputs.size(dim=1) == 4:
                pred = outputs.softmax(1).argmax(1).cpu()
                preds = np.append(preds, pred)
            true = np.append(true, labels.cpu())

    figures = {}
    reports = {}
    if outputs.size(dim=1) == 1:
        preds = scores >= 0.5
        roc_fig, best_threshold, roc_auc = roc_curve_plot(true, scores, True)
        figures['roc'] = roc_fig

    cm = confusion_matrix(true, preds)
    cm = ConfusionMatrixDisplay(cm)
    cm.plot()
    figures['cm_th_05'] = cm.figure_
    reports['th_05'] = classification_report(true, preds,
                                             target_names=class_names,
                                             output_dict=False)

    if outputs.size(dim=1) == 1:
        preds = scores >= best_threshold
        cm = confusion_matrix(true, preds)
        cm = ConfusionMatrixDisplay(cm)
        cm.plot()
        figures['cm_th_best'] = cm.figure_
        reports['th_best'] = classification_report(true, preds,
                                                   target_names=class_names,
                                                   output_dict=False)

    return reports, figures, best_threshold, roc_auc


def roc_curve_plot(true, scores: float, show: bool):

    fpr, tpr, thresholds = roc_curve(true, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", lw=lw,
             label="ROC curve (area = %0.2f)" % roc_auc,)

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()

    idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[idx]
    print('Best threshold = ', best_threshold)

    return fig, thresholds[idx], roc_auc
