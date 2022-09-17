import time
import copy
import torch

import numpy as np
# from deactivate_batchnorm import deactivate_batchnorm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from my_drawings import my_show_image


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
    patience = 150
    accuracy_stats = {"train": [], "val": []}
    loss_stats = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1 :3}/{num_epochs}', end=' ')

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


def test_net(net, data_loaders: dict, class_names: list, device,
             drawing_num=0):

    net.eval()
    preds = np.array([])
    true = np.array([])
    positive_age = np.array([])
    negative_age = np.array([])
    drawings = 0
    for batches in data_loaders["test"]:

        images = batches[0].to(device)
        targets = batches[1]['labels'].to(device)

        with torch.no_grad():
            outputs = net(images)
            if len(outputs) > 1:
                outputs = outputs[0]
            if outputs.size(dim=1) == 1:
                pred = torch.sigmoid(
                    outputs).reshape(-1).detach().cpu().numpy().round()
                preds = np.append(preds, pred)
            elif outputs.size(dim=1) == 4:
                pred = outputs.softmax(1).argmax(1).cpu()
                preds = np.append(preds, pred)
            true = np.append(true, targets.cpu())
            # ---------
            # print("PRED: ", pred.item(), " TRUE: ", targets.cpu().item(),
            #       " age: ", batches[1]['age'].item(),
            #       ' eq ', pred.item() == targets.item())

            if pred.item() == targets.item():
                positive_age = np.append(positive_age,
                                         batches[1]['age'].item())
            else:
                negative_age = np.append(negative_age,
                                         batches[1]['age'].item())
            # ---------

            #  plot images and predictions
            for i in range(len(images)):
                if drawings == drawing_num:
                    break
                else:
                    dcm = [batches[0][i], {'view': batches[1]['view'][i],
                                           'class': batches[1]['class'][i]}]
                    prediction = int(preds[drawings])
                    my_show_image(dcm, with_marks=True,
                                  prediction=prediction,
                                  class_names=class_names)
                    drawings += 1

    cm = confusion_matrix(true, preds)
    cm = ConfusionMatrixDisplay(cm)
    cm.plot()
    if 0:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        print(classification_report(true, preds, target_names=class_names))
    # print("Average positive - age ", np.mean(positive_age))
    # print("Average negative - age ", np.mean(negative_age))

    report = classification_report(true, preds, target_names=class_names,
                                   output_dict=False)

    return report, cm.figure_
