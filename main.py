# %%
import torch
import os
from torchvision import transforms
from my_drawings import my_show_image, my_show_training_results
from my_stats import make_df, show_stats
from BreastCancerDataset import BreastCancerDataset
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from net_utils import train_net, test_net
from gen_data_loader import gen_data_loader
from random_split_df import random_split_df

# ------------
# from pydicom import dcmread
# os.chdir(os.path.join(root, 'Malignant'))
# ds = dcmread(os.listdir(os.getcwd())[1])
# ------------

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# %%
view = ['CC', 'MLO']  # None
class_names = ['Normal', 'Benign', 'Malignant', 'Lymph_nodes']
class_names = ['No cancer', 'Cancer']

root = '/media/dysk/student2/mammografia/Mammografie'
file_dir = '/media/dysk/student2/mammografia/Zapisy/stats_pickle'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = None  # normalize

df = make_df(root, from_file=True, save_to_file=False, file_dir=file_dir)
# data = df.to_dict('records')  # list of patients as dicts
# %%
train_set, val_set, test_set = random_split_df(df=df, seed=50)
data_loaders, data_loaders_sizes = gen_data_loader(root,
                                                   train_set,
                                                   val_set,
                                                   test_set,
                                                   'CC',
                                                   transforms=None)

# %%
# make_df(cc_list)
# if 0:
#     for i in range(10):
#         my_show_image(normal_cc[i], with_marks=True)
#     for i in range(10):
#         my_show_image(benign_cc[i], with_marks=True)
#     for i in range(10):
#         my_show_image(malignant_cc[i], with_marks=True)
#     for i in range(10):
#         my_show_image(lymph_nodes_cc[i], with_marks=True)
# if 0:
#     show_stats(cc_list)

# %%

# %%
net = models.resnet18(pretrained=True)
num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 1)  # dla BCE 1, dla CCE len(class_names)
net = net.to(device)
# %%
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9,
                      weight_decay=0.001)
# Decay LR by a factor of 0.1 every x epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# %% ---------------------------------------------------------
net, loss_stats, accuracy_stats = train_net(net, data_loaders,
                                            data_loaders_sizes,
                                            device, criterion,
                                            optimizer, exp_lr_scheduler,
                                            num_epochs=200)
# %% tymczasowe wykresy jak sie policzy
my_show_training_results(accuracy_stats, loss_stats)

# %% lr finder, learner fastai, metrics sckitlearn
test_net(net, data_loaders, class_names, device)
# %%
model_save_path = (
    '/media/dysk/student2/mammografia/'
    + 'Zapisane_modele/r18_13_07/resnet18_2cl_xx_xx')
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
test_net(net_loaded, data_loaders, class_names, device)
# %%
new_net = models.resnet18(pretrained=True)
num_features = new_net.fc.in_features

ct = 0
for child in new_net.children():
    ct += 1
    if ct < 3:
        for param in child.parameters():
            param.requires_grad = False

new_net.fc = nn.Linear(num_features, 1)
new_net.to(device)
# %%
