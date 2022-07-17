# %%
import torch
from torchvision import transforms
from my_drawings import my_show_image, my_show_training_results
from my_stats import make_df, show_stats
from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import nn
from net_utils import train_net, test_net
from gen_data_loader import gen_data_loader
from random_split_df import random_split_df
from LRFinder import LRFinder, plot_lr_finder

# ------------
# import os
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
transform = transforms.Compose([normalize, transforms.Resize((224, 224))])

transform = None
df = make_df(root, from_file=True, save_to_file=False, file_dir=file_dir)
# data = df.to_dict('records')  # list of patients as dicts
# %%
train_set, val_set, test_set = random_split_df(df=df, seed=50)
data_loaders, data_loaders_sizes, ds = gen_data_loader(root,
                                                       train_set,
                                                       val_set,
                                                       test_set,
                                                       'CC',
                                                       transforms=transform,
                                                       batch_size=16)

# %%
if 1:
    for i in range(1):
        my_show_image(ds['train'][i], with_marks=True)

# if 0:
#     show_stats(cc_list)

# %%

# %%
net = models.resnet18(pretrained=True)

num_features = net.fc.in_features
net.fc = nn.Linear(num_features, 1)  # dla BCE 1, dla CCE len(class_names)
net = net.to(device)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))

# print(f'The model has {count_parameters(net):,} trainable parameters')

# %%

START_LR = 1e-7
optimizer = optim.Adam(net.parameters(), lr=START_LR)
criterion = nn.BCELoss()
criterion = criterion.to(device)
END_LR = 1
NUM_ITER = 300

if 1:
    lr_finder = LRFinder(net, optimizer, criterion, device)
    lrs, losses = lr_finder.range_test(data_loaders['train'], END_LR, NUM_ITER)
    plot_lr_finder(lrs, losses, skip_start=0, skip_end=0)

FOUND_LR = 0  # 2e-4 for adamoptim resnet50 bs32 resize224

params = [
          {'params': net.conv1.parameters(), 'lr': FOUND_LR / 10},
          {'params': net.bn1.parameters(), 'lr': FOUND_LR / 10},
          {'params': net.layer1.parameters(), 'lr': FOUND_LR / 8},
          {'params': net.layer2.parameters(), 'lr': FOUND_LR / 6},
          {'params': net.layer3.parameters(), 'lr': FOUND_LR / 4},
          {'params': net.layer4.parameters(), 'lr': FOUND_LR / 2},
          {'params': net.fc.parameters()}
         ]

optimizer = optimizer = optim.SGD(params, lr=FOUND_LR,
                                  momentum=0.9, weight_decay=0.001)

EPOCHS = 10
STEPS_PER_EPOCH = data_loaders_sizes['train']
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH

MAX_LRS = [p['lr'] for p in optimizer.param_groups]

scheduler = lr_scheduler.OneCycleLR(optimizer,
                                    max_lr=MAX_LRS,
                                    total_steps=TOTAL_STEPS)
# %% ---------------------------------------------------------
net, loss_stats, accuracy_stats = train_net(net, data_loaders,
                                            data_loaders_sizes,
                                            device, criterion,
                                            optimizer, scheduler,
                                            num_epochs=EPOCHS)
# %%
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
