# %%
import torch
import torch.nn as nn
from torchvision import transforms as T
from my_drawings import my_show_image, my_show_training_results
from my_stats import make_df, show_stats
from choose_NCO import choose_NCO
from torchvision import models
from torch.optim import lr_scheduler
from net_utils import train_net, test_net
from gen_data_loader import gen_data_loader
from random_split_df import random_split_df
from LRFinder import LRFinder, plot_lr_finder
from get_points_to_crop import get_points
# ------------
# import os
# from pydicom import dcmread
# os.chdir(os.path.join(root, 'Malignant'))
# ds = dcmread(os.listdir(os.getcwd())[1])
# ------------

# %%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
view = ['CC', 'MLO']  # None
root = '/media/dysk/student2/mammografia/Mammografie'
file_dir = '/media/dysk/student2/mammografia/Zapisy/stats_pickle'

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

transform = T.Compose([
                    #    T.Resize((224, 224)),
                       T.RandomAffine(degrees=(3), translate=(0, 0.1)),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip()])
transforms_val_test = None #T.Compose([T.Resize((224, 224))])

tfs = [transform, transforms_val_test, transforms_val_test]
df = make_df(root, from_file=True, save_to_file=False, file_dir=file_dir)
# data = df.to_dict('records')  # list of patients as dicts
# %%
train_set, val_set, test_set = random_split_df(df=df, seed=50)
data_loaders, data_loaders_sizes, ds = gen_data_loader(root,
                                                       train_set,
                                                       val_set,
                                                       test_set,
                                                       view,
                                                       transforms=tfs,
                                                       batch_size=4)

# %%
if 0:  # find column to crop
    get_points(ds, 222)  # crop inny dla MLO moze i jest dla odwroconych
if 1:
    for i in range(1):
        my_show_image(ds['val'][i], with_marks=True)
# if 0:
#     show_stats(cc_list)

# %%
if 0:
    for wd in [1e-3]:  # , 2e-2, 3e-3, 1e-3]:
        START_LR = 1e-5
        net, criterion, optimizer = choose_NCO(net_ar='resnet50',
                                               device=device,
                                               criterion_type='bce',
                                               optimizer_type='sgd',
                                               lr=START_LR, wd=wd)
        END_LR = 3
        NUM_ITER = 30
        lr_finder = LRFinder(net, optimizer, criterion, device)
        lrs, losses = lr_finder.range_test(data_loaders['val'],
                                           END_LR, NUM_ITER)
        print('WD:', wd)
        plot_lr_finder(lrs, losses, skip_start=0, skip_end=0)

# %%
FOUND_LR = 3e-3
# params = [
#           {'params': net.conv1.parameters(), 'lr': FOUND_LR / 10},
#           {'params': net.bn1.parameters(), 'lr': FOUND_LR / 10},
#           {'params': net.layer1.parameters(), 'lr': FOUND_LR / 8},
#           {'params': net.layer2.parameters(), 'lr': FOUND_LR / 6},
#           {'params': net.layer3.parameters(), 'lr': FOUND_LR / 4},
#           {'params': net.layer4.parameters(), 'lr': FOUND_LR / 2},
#           {'params': net.fc.parameters()}
#          ]
# EPOCHS = 100
# STEPS_PER_EPOCH = len(data_loaders['train'])
# TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
# MAX_LRS = [p['lr'] for p in optimizer.param_groups]
# scheduler = lr_scheduler.OneCycleLR(optimizer,
#                                     max_lr=MAX_LRS,
#                                     total_steps=TOTAL_STEPS)
# %%
EPOCHS = 100
# lr = 1e-2
lr = FOUND_LR
wd = 3e-3
net, criterion, optimizer = choose_NCO(net_ar='resnet18',
                                       device=device,
                                       criterion_type='bce',
                                       optimizer_type='sgd',
                                       lr=lr, wd=wd)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
# % ---------------------------------------------------------
net, loss_stats, accuracy_stats = train_net(net, data_loaders,
                                            data_loaders_sizes,
                                            device, criterion,
                                            optimizer, scheduler,
                                            num_epochs=EPOCHS)
my_show_training_results(accuracy_stats, loss_stats)
# %% lr finder, learner fastai, metrics sckitlearn
class_names = ['Normal', 'Benign', 'Malignant', 'Lymph_nodes']
class_names = ['No cancer', 'Cancer']
# class_names = ['Normal', 'Benign', 'Malignant']
test_net(net, data_loaders, class_names, device, drawing_num=15)
# %%
model_save_path = (
    '/media/dysk/student2/mammografia/Zapisy/r50/resnet50')


# %%
if 0:
    torch.save(net.state_dict(), model_save_path)
# %%
model_load_path = (
    '/media/dysk/student2/mammografia/'
    + 'Zapisane_modele/r18_13_07/resnet18_2cl_13_07')
net_loaded = models.resnet18()
num_features = net_loaded.fc.in_features
net_loaded.fc = nn.Linear(num_features, 1)
net_loaded = net_loaded.to(device)
net_loaded.load_state_dict(torch.load(model_load_path))
# %%
test_net(net_loaded, data_loaders, class_names, device,
         drawing_num=1)

# %%
