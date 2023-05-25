# %%
import pandas as pd
import torch
from torchvision import transforms as T
from my_drawings import my_show_image, my_show_training_results
from my_stats import make_df, make_img_stats_dict, box_plot
from choose_NCOS import choose_NCOS
from net_utils import train_net, test_net
from gen_data_loader import gen_data_loader
from random_split_df import random_split_df
from LRFinder import LRFinder, plot_lr_finder
from get_points_to_crop import get_points
from tile_maker import get_tiles
import os
import numpy as np

import matplotlib.pyplot as plt  # temp
# from tile import convert_img_to_bag, get_tiles  # temp

import argparse
import yaml
import neptune.new as neptune
import uuid
import copy
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
# %%
# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
def get_args_parser():
    default = '/home/jr_buler/Projekt_MMG/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=default,
                        help=help)
    return parser


parser = get_args_parser()
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
# %%
# SET PARAMS------------------------------------------------------------------
selected_device = config['device'][0]
device = torch.device(selected_device if torch.cuda.is_available() else "cpu")

seed = config['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataloader_path = config['data_sets']['dataloader_path']
dataloader_sizes_path = config['data_sets']['dataloader_sizes']

create_new_dl = config['data_sets']['create_new_dl']
batch_size = config['training_plan']['parameters']['batch_size']
num_workers = config['data_sets']['num_workers']
view = config['data_sets']['image_view']
class_names = config['data_sets']['class_names']

root = config['data_sets']['root_dir']
file_dir = config['data_sets']['file_dir']
train_rest_frac = config['data_sets']['split_fraction_train_rest']
val_test_frac = config['data_sets']['split_fraction_val_test']
bag_size_train = config['data_sets']['bag_size_train']
bag_size_val_test = config['data_sets']['bag_size_val_test']

image_size = config['image']['size']
image_multimodality = config['image']['multimodal']

patch_size = config['data_sets']['patch_size']
overlap_train = config['data_sets']['overlap_train_val']
overlap_val_test = config['data_sets']['overlap_val_test']

# %%

# TRANSFORMS------------------------------------------------------------------
cj_prob = 0.5
cj_bright = 0.25
cj_contrast = 0.25
cj_sat = 0.25
cj_hue = 0.25
gaussian_blur_prob = 0.5

input_size = patch_size
min_scale = 0.8

color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)
gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

transform = T.Compose([  # T.RandomAffine(degrees=(0), translate=(0, 0.1)),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       T.RandomResizedCrop(size=input_size,
                                           scale=(min_scale, 1.0),
                                           antialias=True),
                       # T.RandomApply([color_jitter], p=cj_prob),
                       # T.RandomApply([gaussian_blur], p=gaussian_blur_prob)
                       ])

transforms_val_test = None
tfs = [transform, transforms_val_test, transforms_val_test]

# GET TILES (PATCHES) COORDS (size/overlap)-----------------------------------
tiles_train = get_tiles(image_size[0], image_size[1],
                        patch_size, patch_size,
                        overlap_train)
tiles_test_val = get_tiles(image_size[0], image_size[1],
                           patch_size, patch_size,
                           overlap_val_test)
tiles = [tiles_train, tiles_test_val]

# tiles_scale_1 = get_tiles(3518, 2800,
#                           patch_size, patch_size,
#                           overlap_train)
# tiles_scale_2 = get_tiles(3518, 2800,
#                           patch_size*2, patch_size*2,
#                           overlap_val_test)
# tiles = [tiles_scale_1, tiles_scale_2]
# tiles = [tiles, tiles]
# %%
# MAKE & SAVE NEW DATASET OR LOAD CURRENTLY SAVED ONE-------------------------
df = make_df(root,
             from_file=True,
             save_to_file=False,
             file_dir=file_dir)


# ***************
# os.chdir('/media/dysk/student2/CMMD/TheChineseMammographyDatabase/')
# df = pd.read_csv('CMMD_clinicaldata_revision_CSV.csv', sep=';')
# df = df.sort_values(
#     by=['LeftRight']).groupby('ID1').agg({"Age": np.mean,
#                                           "LeftRight": list,
#                                           "classification": list,
#                                           }).reset_index()
# root = '/media/dysk/student2/CMMD/TheChineseMammographyDatabase/CMMD/'
# ***************

# %%

if create_new_dl:
    train_set, val_set, test_set = random_split_df(df,
                                                   train_rest_frac,
                                                   val_test_frac,
                                                   seed=seed)

    data_loaders, data_loaders_sizes, ds = gen_data_loader(
        root=root,
        train_df=train_set,
        val_df=val_set,
        test_df=test_set,
        view=view,
        transforms=tfs,
        batch_size=batch_size,
        nw=num_workers,
        conv_to_bag=True,
        bag_size_train=bag_size_train,
        bag_size_val_test=bag_size_val_test,
        tiles=tiles,
        img_size=image_size,
        is_multimodal=image_multimodality
        )
    torch.save(data_loaders, dataloader_path)
    torch.save(data_loaders_sizes, dataloader_sizes_path)
else:
    data_loaders = torch.load(dataloader_path)
    data_loaders_sizes = torch.load(dataloader_sizes_path)

if 0:  # find column to crop
    get_points(ds, 222)
if 0:
    for i in range(1681):
        my_show_image(ds['test'][i], with_marks=True)
# %%
# IMAGE STATS - MEAN - STD - VAR - AREA
if 0:
    stats_dict = make_img_stats_dict(ds)
if 0:
    box_plot(stats_dict, k='area')
# %%
if 0:
    for i in range(30):
        i = i+30
        if ds['test'][i][1]['class'] == 'Malignant':
            my_show_image(ds['test'][i], with_marks=True, format_type='svg')
# %%
if 0:
    from pydicom import dcmread
    dir = root
    os.chdir(dir)
    for folder in os.listdir(os.getcwd()):
        os.chdir(os.path.join(dir, folder))
        for file in os.listdir(os.getcwd()):
            dcm = dcmread(file)
            if dcm.ManufacturerModelName != 'Mammomat Inspiration':
                print(dcm.ManufacturerModelName)
                print(folder, end='/')
                print(file)

            # print(dcm.AcquisitionDate, end=' ')
            # print(dcm.WindowCenter, end=' ')
            # print(dcm.WindowWidth, end=' ')
            # print(dcm.RescaleSlope, end=' ')
            # print(dcm.RescaleIntercept)
            # print(dcm.BodyPartThickness)
# %%
# LEARNING RATE FINDER FOR ONE-CYCLE POLICY-----------------------------------
if 0:
    for wd in [1e-4]:  # , 2e-3, 3e-4, 1e-5]:
        START_LR = 1e-6
        net, criterion, optimizer = choose_NCOS(net_ar='gmil',
                                                device=device,
                                                pretrained=True,
                                                criterion_type='bce',
                                                optimizer_type='sgd',
                                                lr=START_LR, wd=wd)
        END_LR = 1
        NUM_ITER = 1
        lr_finder = LRFinder(net, optimizer, criterion, device)
        lrs, losses = lr_finder.range_test(data_loaders['val'],
                                           END_LR, NUM_ITER)
        print('WD:', wd)
        plot_lr_finder(lrs, losses, skip_start=0, skip_end=0)

# %%
# SET TRAINING PLAN-----------------------------------------------------------
EPOCHS = config['training_plan']['parameters']['epochs'][0]
lr = config['training_plan']['parameters']['lr'][0]
wd = config['training_plan']['parameters']['wd'][0]
grad_accu = config['training_plan']['parameters']['grad_accu']['is_on']
grad_accu_steps = config['training_plan']['parameters']['grad_accu']['steps']
net_ar = config['training_plan']['architectures']['names']
net_ar_dropout = config['training_plan']['architectures']['dropout']
criterion_type = config['training_plan']['criterion']
optimizer_type = config['training_plan']['optim_name']
scheduler_type = config['training_plan']['scheduler']

if scheduler_type == 'one-cycle':
    scheduler_type['epochs'] = EPOCHS
    scheduler_type['steps_per_epoch'] = len(data_loaders['train'])

net, criterion, optimizer, scheduler = choose_NCOS(
    net_ar=net_ar,
    dropout=net_ar_dropout,
    device=device,
    pretrained=True,
    criterion_type=criterion_type,
    optimizer_type=optimizer_type,
    lr=lr, wd=wd,
    scheduler=scheduler_type)

# %%
# LOAD CLR FEATURE  EXTRACTOR / DEACT BACTH NORM------------------------------
if 1:
    from torch import nn

    def deactivate_batchnorm(net):
        if isinstance(net, nn.BatchNorm2d):
            net.track_running_stats = False
            net.running_mean = None
            net.running_var = None
            # net.momentum = 0.01

    # def change_bn_to_gn(net):
    #     for name, module in net.named_modules():
    #         if isinstance(module, nn.BatchNorm2d):
    #             # Get current bn layer
    #             bn = getattr(net, name)
    #             # Create new gn layer
    #             gn = nn.GroupNorm(1, bn.num_features)
    #             # Assign gn
    #             print('Swapping {} with {}'.format(bn, gn))
    #             setattr(net, name, gn)

    net.apply(deactivate_batchnorm)
if 0:
    fn = '966ccf26-e250-4ea8-a5f5-9fa7fc9287d8'
    model_load_path = ('/media/dysk_a/jr_buler/mammografia/Zapisy/'
                       + 'neptune_saved_models/' + fn)
    net_sd = torch.load(model_load_path, map_location=device)
    net_sd = copy.deepcopy(net.state_dict())

    net.load_state_dict(net_sd)
    net.to(device)
# %%
# TRAIN NETWORK---------------------------------------------------------------
if 1:
    run = neptune.init(project='ProjektMMG/Mammografia')
    run['config'] = config
else:
    run = None

state_dict_BL, state_dict_BACC, loss_stats, accuracy_stats = train_net(
    net, data_loaders,
    data_loaders_sizes,
    device, criterion,
    optimizer, scheduler,
    num_epochs=EPOCHS,
    neptune_run=run,
    grad_acc_mode=grad_accu,
    accum_steps=grad_accu_steps
    )

if 0:
    my_show_training_results(accuracy_stats, loss_stats)
# %%
# TEST NETWORK----------------------------------------------------------------
if 1:
    net_BACC = net
    net_BACC.load_state_dict(state_dict_BACC)
    metrics, figures, best_th, roc_auc = test_net(net_BACC, data_loaders,
                                                  class_names, device)
    unique_filename1 = str(uuid.uuid4())
    model_save_path = ('/media/dysk/student2/mammografia/Zapisy/'
                       + 'neptune_saved_models/' + unique_filename1)
    if run is not None:
        run['test/BACC/metrics'].log(metrics['th_05'])
        run['test/BACC/conf_mx'].upload(figures['cm_th_05'])
        run['test/BACC/auc_roc'] = roc_auc
        if criterion_type == 'bce':
            run['test/BACC/th_best'] = best_th
            run['test/BACC/metrics_th_best'].log(metrics['th_best'])
            run['test/BACC/conf_mx_th_best'].upload(figures['cm_th_best'])
            run['test/BACC/roc'].upload(figures['roc'])
        run['test/BACC/file_name'] = unique_filename1

    torch.save(net_BACC.state_dict(), model_save_path)
    del net_BACC

if 1:
    net_BL = net
    net_BL.load_state_dict(state_dict_BL)
    metrics, figures, best_th, roc_auc = test_net(net_BL, data_loaders,
                                                  class_names, device)
    unique_filename2 = str(uuid.uuid4())
    model_save_path = ('/media/dysk/student2/mammografia/Zapisy/'
                       + 'neptune_saved_models/' + unique_filename2)
    if run is not None:
        run['test/BL/metrics'].log(metrics['th_05'])
        run['test/BL/conf_mx'].upload(figures['cm_th_05'])
        if criterion_type == 'bce':
            run['test/BL/th_best'] = best_th
            run['test/BL/metrics_th_best'].log(metrics['th_best'])
            run['test/BL/conf_mx_th_best'].upload(figures['cm_th_best'])
            run['test/BL/roc'].upload(figures['roc'])
        run['test/BL/file_name'] = unique_filename2

    torch.save(net_BL.state_dict(), model_save_path)
    del net_BL


if run is not None:
    run['test/auc_roc'] = roc_auc
    run.stop()
# %%

# psum  = torch.tensor([0.0, 0.0, 0.0])
# psum_sq = torch.tensor([0.0, 0.0, 0.0])
# tiles_count = 0

# # dataloader: B N c h w
# # bag:          N c h w
# # tile:           c h w

# for batch in data_loaders['test']:
#     for bag in batch[0]:
#         for tile in bag:
#             tiles_count += 1
#             psum    += tile.sum(axis=[1,2])
#             psum_sq += (tile ** 2).sum(axis=[1,2])

# image_size = tiles[0][0][2]
# count = tiles_count * image_size * image_size

# # mean and std
# total_mean = psum / count
# total_var  = (psum_sq / count) - (total_mean ** 2)
# total_std  = torch.sqrt(total_var)

# # output
# print('mean: '  + str(total_mean))
# print('std:  '  + str(total_std))
# %%
# Rysowanie hist dla setow val i test
# confidence dla kazdego pacjenta w zbiorze

# 34a2036b-0f9c-4dc5-b583-86ec05c932e3  best rocacc
# eaedf20a-8cac-479f-a5f3-07727bf3b726  best roclss
if 0:

    std = torch.load('/media/dysk/student2/mammografia/Zapisy/'
                     + 'neptune_saved_models/'
                     + '34a2036b-0f9c-4dc5-b583-86ec05c932e3',
                     map_location=device)
    net.load_state_dict(std)
    phases = ['val', 'test']
    net.to(device)
    net.eval()

    ids_list = list()
    pred_list = list()
    class_list = list()
    for phase in phases:
        for images, targets in data_loaders[phase]:
            images = images.to(device)
            labels = targets["labels"].to(device)
            with torch.no_grad():
                outputs = net(images)
                preds = torch.sigmoid(
                    outputs[0]).reshape(-1).detach().cpu()

                if preds.numpy().round() == labels.item():
                    ids_list.append(targets['patient_id'])
                    pred_list.append(preds.item())
                    class_list.append(targets['class'])
    pred_df = pd.DataFrame()
    pred_df['scores'] = pred_list
    pred_df['class'] = class_list
    pred_df['id'] = ids_list
    pred_df['scores'].plot.hist(grid=True, bins=20, rwidth=0.9,
                                color='#607c8e')
    plt.title('Correct preds')
    plt.xlabel('Scores')
    plt.ylabel('Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    ids_list = list()
    pred_list = list()
    class_list = list()
    laterality_list = list()
    for phase in phases:
        for images, targets in data_loaders[phase]:
            images = images.to(device)
            labels = targets["labels"].to(device)
            with torch.no_grad():
                outputs = net(images)
                preds = torch.sigmoid(
                    outputs[0]).reshape(-1).detach().cpu()

                if preds.numpy().round() != labels.item():
                    ids_list.append(targets['patient_id'])
                    pred_list.append(preds.item())
                    class_list.append(targets['class'])
                    laterality_list.append(targets['laterality'])
    pred_df = pd.DataFrame()
    pred_df['scores'] = pred_list
    pred_df['class'] = class_list
    pred_df['id'] = ids_list
    pred_df['scores'].plot.hist(grid=True, bins=20, rwidth=0.9,
                                color='#607c8e')
    print("False negatives")
    print(pred_df[pred_df['scores'] < 0.2])
    print("False positives")
    print(pred_df[pred_df['scores'] > 0.8])
    plt.title('Incorrect preds')
    plt.xlabel('Scores')
    plt.ylabel('Counts')
    plt.grid(axis='y', alpha=0.75)
    plt.show()
# %%
if 0:

    phases = ['train', 'val', 'test']
    num_instances = list()
    for phase in phases:
        for images, targets in data_loaders[phase]:
            num_instances.append(len(images[0]))
        df = pd.DataFrame()
        df['num_instances'] = num_instances
        df['num_instances'].plot.hist(grid=True, bins=20, rwidth=0.9,
                                      color='#607c8e')
        plt.title('Bag sizes' + ' [' + phase + ']')
        plt.xlabel('Number of instances')
        plt.ylabel('Bags')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
# %%
