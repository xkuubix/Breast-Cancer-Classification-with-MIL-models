# %%
import os
import yaml
import uuid
import copy
import argparse
import warnings
import numpy as np
import pandas as pd
import neptune.new as neptune

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
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

dataloader_path = config['data_sets']['dataloader_path']
dataloader_sizes_path = config['data_sets']['dataloader_sizes']

dataset = config['data_sets']['dataset']
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

not_multiscale = config['data_sets']['not_multiscale']
patch_size = config['data_sets']['patch_size']
overlap_train = config['data_sets']['overlap_train_val']
overlap_val_test = config['data_sets']['overlap_val_test']


# %%
# TRANSFORMS------------------------------------------------------------------
input_size = patch_size
min_scale = 0.8

gaussian_blur = T.GaussianBlur(kernel_size=5)
color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)

# Definiuj Random Apply
augmentation_rotation = T.RandomApply([T.RandomRotation(degrees=(90, 90))], p=0.5)
augmentation_gblur = T.RandomApply([gaussian_blur], p=0.5)
augmentation_cjitter = T.RandomApply([color_jitter], p=0.5)
augmentation_rrcrop = T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0),
                                         antialias=True)



                       
transform = T.Compose([augmentation_rotation,
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       augmentation_cjitter,
                       augmentation_gblur,
                       augmentation_rrcrop])
# transform = None
transforms_val_test = None
tfs = [transform, transforms_val_test, transforms_val_test]

# %%
# GET TILES (PATCHES) COORDS (size/overlap)-----------------------------------
if not_multiscale:
    tiles_train = get_tiles(image_size[0], image_size[1],
                            patch_size, patch_size,
                            overlap_train)
    tiles_test_val = get_tiles(image_size[0], image_size[1],
                            patch_size, patch_size,
                            overlap_val_test)
    tiles = [tiles_train, tiles_test_val]
else:
    tiles_scale_1 = get_tiles(image_size[0], image_size[1],
                            patch_size, patch_size,
                            overlap_train)
    tiles_scale_2 = get_tiles(image_size[0], image_size[1],
                            patch_size*2, patch_size*2,
                            overlap_val_test)
    tiles = [tiles_scale_1, tiles_scale_2]
    tiles = [tiles, tiles]

# %%
# MAKE & SAVE NEW DATASET OR LOAD CURRENTLY SAVED ONE-------------------------
if dataset == 'MUG':
    df = make_df(root,
                from_file=True,
                save_to_file=False,
                file_dir=file_dir)
    # ***************
elif dataset == 'CMMD':
    os.chdir('/media/dysk_a/jr_buler/TheChineseMammographyDatabase/')
    df = pd.read_csv('CMMD_clinicaldata_revision_CSV.csv', sep=';')
    df = df.sort_values(
        by=['LeftRight']).groupby('ID1').agg({"Age": np.mean,
                                            "LeftRight": list,
                                            "classification": list,
                                            }).reset_index()
    root = '/media/dysk_a/jr_buler/TheChineseMammographyDatabase/CMMD/'
# ***************

# %%
if create_new_dl:
    train_set, val_set, test_set = random_split_df(df,
                                                   train_rest_frac,
                                                   val_test_frac,
                                                   seed=seed)

    data_loaders, data_loaders_sizes, ds = gen_data_loader(
        root=root,
        dataset=dataset,
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
    net.apply(deactivate_batchnorm)
if 0:
    fn = 'd9c349b4-b2db-437b-8563-86661e3850ce'
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
    run['transforms'] = transform
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

# %%
# TEST NETWORK----------------------------------------------------------------
if 1:
    net_BACC = net
    net_BACC.load_state_dict(state_dict_BACC)
    metrics, figures, best_th, roc_auc = test_net(net_BACC, data_loaders,
                                                  class_names, device)
    unique_filename1 = str(uuid.uuid4())
    model_save_path = ('/media/dysk_a/jr_buler/mammografia/Zapisy/'
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

    net_BL = net
    net_BL.load_state_dict(state_dict_BL)
    metrics, figures, best_th, roc_auc = test_net(net_BL, data_loaders,
                                                  class_names, device)
    unique_filename2 = str(uuid.uuid4())
    model_save_path = ('/media/dysk_a/jr_buler/mammografia/Zapisy/'
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
else:
    print('run not set')
