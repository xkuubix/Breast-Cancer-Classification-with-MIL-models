# %%
import torch
import argparse
import yaml
import neptune.new as neptune
import uuid
import copy
from torchvision import transforms as T
from choose_NCOS import choose_NCOS
from my_stats import make_df
from net_utils import train_net, test_net
from gen_data_loader import gen_data_loader
from tile_maker import get_tiles
from sklearn.model_selection import KFold
from torch import nn
# import pandas as pd
# import numpy as np
# import os


def deactivate_batchnorm(net):
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None
        # net.momentum = 0.01


def print_ds_info(ds, name_to_print, view):
    count_classes = {'Normal': 0, 'Benign': 0,
                     'Malignant': 0, 'Lymph_nodes': 0}
    for _, target in ds:
        for k in count_classes.keys():
            if target['class'] == k:
                count_classes[k] += 1
    print('\n' + name_to_print, view, 'images')
    for k in count_classes.keys():
        print(k, count_classes[k])


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
# TRANSFORMS------------------------------------------------------------------
input_size = patch_size
min_scale = 0.8

transform = T.Compose([  # T.RandomAffine(degrees=(0), translate=(0, 0.1)),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       T.RandomResizedCrop(size=input_size,
                                           scale=(min_scale, 1.0),
                                           antialias=True)
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
# %%
# MAKE & SAVE NEW DATASET OR LOAD CURRENTLY SAVED ONE-------------------------
df = make_df(root,
             from_file=False,
             save_to_file=False,
             file_dir=file_dir)

# **********
#  TUTAJ ZMIENIC SCIEZKI POZNIEN 
# os.chdir('/media/dysk/student2/CMMD/TheChineseMammographyDatabase/')
# df = pd.read_csv('CMMD_clinicaldata_revision_CSV.csv', sep=';')
# df = df.sort_values(
#     by=['LeftRight']).groupby('ID1').agg({"Age": np.mean,
#                                           "LeftRight": list,
#                                           "classification": list,
#                                           }).reset_index()
# # df = df[df['ID1'].str.contains('D2-') == False]
# root = '/media/dysk/student2/CMMD/TheChineseMammographyDatabase/CMMD/'
# **********

# %%
# Net
# SET TRAINING PLAN--------------------------------------------------------
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

net, criterion, optimizer, scheduler = choose_NCOS(
    net_ar=net_ar,
    dropout=net_ar_dropout,
    device=device,
    pretrained=True,
    criterion_type=criterion_type,
    optimizer_type=optimizer_type,
    lr=lr, wd=wd,
    scheduler=scheduler_type)

# LOAD CLR FEATURE  EXTRACTOR / DEACT BACTH NORM---------------------------
net.apply(deactivate_batchnorm)

net_sd = copy.deepcopy(net.state_dict())
if 1:
    std = torch.load('/media/dysk_a/jr_buler/mammografia/Zapisy/'
                     + 'neptune_saved_models/'
                     + 'ea8571f2-7de0-479c-86b9-261265fb2d52',
                     map_location=device)

# Cross-val splits
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

unique_id = str(uuid.uuid4())
for fold, (train_val_idx, test_idx) in enumerate(kf.split(df)):
    print('------------fold no---------{}----------------------'.format(fold))
    # print('Test indexes: ', test_idx)

    train_val_df = df.drop(index=test_idx)
    test_df = df.drop(index=train_val_idx)
    train_df = train_val_df.sample(frac=0.875, random_state=seed)
    val_df = train_val_df.drop(train_df.index)
    print(len(train_df), len(val_df), len(test_df))
    print('Train')
    print(train_df['class'].value_counts())

    print()
    print('Val')
    print(val_df['class'].value_counts())

    print()
    print('Test')
    print(test_df['class'].value_counts())
    # print(len(train_df[train_df['classification']]))
    continue
    data_loaders, data_loaders_sizes, ds = gen_data_loader(
        root=root,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
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
# %%
    # RESET SEED
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    net, criterion, optimizer, scheduler = choose_NCOS(
        net_ar=net_ar,
        dropout=net_ar_dropout,
        device=device,
        pretrained=True,
        criterion_type=criterion_type,
        optimizer_type=optimizer_type,
        lr=lr, wd=wd,
        scheduler=scheduler_type)
    # LOAD CLR FEATURE  EXTRACTOR / DEACT BACTH NORM---------------------------
    net.apply(deactivate_batchnorm)
    net.load_state_dict(net_sd)
    # net.load_state_dict(std)

    # TRAIN NETWORK------------------------------------------------------------
    if 1:
        run = neptune.init(project='ProjektMMG/Cross-val-nowe-dane')
        run['config'] = config
    else:
        run = None

    run['fold-k'] = fold
    run['unique-id'] = unique_id

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

    if 1:
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
            run['test/auc_roc'] = roc_auc
            if criterion_type == 'bce':
                run['test/BL/th_best'] = best_th
                run['test/BL/metrics_th_best'].log(metrics['th_best'])
                run['test/BL/conf_mx_th_best'].upload(figures['cm_th_best'])
                run['test/BL/roc'].upload(figures['roc'])
            run['test/BL/file_name'] = unique_filename2

        torch.save(net_BL.state_dict(), model_save_path)
        del net_BL

    if run is not None:
        run.stop()
