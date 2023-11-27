# %%
import os
import yaml
import copy
import argparse
import warnings
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from my_stats import make_df
from choose_NCOS import choose_NCOS
from gen_data_loader import gen_data_loader
from random_split_df import random_split_df
from tile_maker import get_tiles

from BreastCancerDataset import CMMD_DS
from BreastCancerDataset import BreastCancerDataset

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
transform = None
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
data_loaders = {}
dl_sizes = {}

if create_new_dl:
    train_set, val_set, test_set = random_split_df(df,
                                                   train_rest_frac,
                                                   val_test_frac,
                                                   seed=seed)
    
    if dataset == 'CMMD':
        test_dataset = CMMD_DS(root, test_set,
                               view, tfs[2],
                               conv_to_bag=True,
                               bag_size=bag_size_val_test,
                               tiles=tiles[1],
                               img_size=image_size,
                               is_multimodal=image_multimodality
                               )
        data_loaders['test'] = DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=num_workers,
                                          collate_fn=None
                                          )
        dl_sizes['test'] = [len(item) for item in data_loaders['test']]

    elif dataset == 'MUG':
        test_dataset = BreastCancerDataset(root, test_set,
                                           view, tfs[2],
                                           conv_to_bag=True,
                                           bag_size=bag_size_val_test,
                                           tiles=tiles[1],
                                           img_size=image_size,
                                           is_multimodal=image_multimodality
                                           )
        data_loaders['test'] = DataLoader(test_dataset, batch_size=1,
                                          shuffle=True, num_workers=num_workers,
                                          collate_fn=None
                                          )

else:
    data_loaders = torch.load(dataloader_path)
    data_loaders_sizes = torch.load(dataloader_sizes_path)

# %%
# SET -----------------------------------------------------------
lr = config['training_plan']['parameters']['lr'][0]
wd = config['training_plan']['parameters']['wd'][0]
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

# %%
# LOAD CLR FEATURE  EXTRACTOR / DEACT BACTH NORM------------------------------

from torch import nn

def deactivate_batchnorm(net):
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None

if 1:
    net.apply(deactivate_batchnorm)
    net_to_load = '1f86047a-92d7-45c0-9b1f-75461227210b'
    std = torch.load('/media/dysk_a/jr_buler/mammografia/Zapisy/'
                     + 'neptune_saved_models/'
                     + net_to_load,
                     map_location=device)
    net.load_state_dict(std)

# %%
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution

def get_attributions(net, input_net, ):
    with torch.no_grad():
        net.zero_grad()
        layer_gc = LayerGradCam(net, net.feature_extractor.layer4[1].conv2)
        attributions = layer_gc.attribute(input_net, target=0).detach().cpu()
        upsampled_attr = LayerAttribution.interpolate(attributions, (224, 224),
                                                    interpolate_mode='nearest')
        return upsampled_attr
    '''
        sprawdzić interpolate_mode = 'nearest', 'linear', area', 'bilinear'
        ponoć linear lepsze od nearest - więcej detali
                    
    '''

with torch.no_grad():
    data = iter(data_loaders['test'])
    d_s = []
    for i in range(40):
        h, w = image_size[0], image_size[1]
        empty_img = torch.zeros(3, h, w )
        empty_img_counts = torch.ones(3, h, w )
        to_show = torch.ones(3, h, w )

        d_s = next(data)

        if d_s[1]['class'][0] in ['Normal', 'Benign'] :
            continue

        im = d_s[1]['full_image'].squeeze(dim=0)
        id = d_s[1]['tiles_indices'].squeeze(dim=0)

        ####### sid = '[' + d_s[1]['patient_id'][0] + ']'
        net.eval()
        input_net = d_s[0].to(device)
        output = net(input_net, None)  # , d_s[1]['tile_cords'])
        score = torch.sigmoid(output[0]).detach().cpu().numpy()
        pred = score.round()
        # _, pred = torch.max(output[0].reshape(-1, 4), 1)
        # if pred != d_s[1]['labels'] or d_s[1]['labels'] == 0:
        #     continue
        

        upsampled_attr = get_attributions(net, input_net, )
    

        upsampled_attr = upsampled_attr.permute(1,0,2,3).squeeze()
        attribution_map = torch.zeros(3, h, w)

        # gmil
        weights = net.A
        # clam
        # weights = output[3]

        weights = weights.detach().cpu().squeeze()

        for item in range(len(id)):
            h_min, w_min, dh, dw, _, _ = tiles[1][id][item]
            to_show[:, h_min:h_min+dh, w_min:w_min+dw] = d_s[0][0][item]

            empty_img[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                weights[item]
            empty_img_counts[:, h_min:h_min+dh, w_min:w_min+dw] += 1
            attribution_map[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                    upsampled_attr[item]

        empty_img = torch.div(empty_img, empty_img_counts)
        empty_img /= torch.max(empty_img.reshape(-1))
        attribution_map = attribution_map.sum(0)
        attribution_map = attribution_map/attribution_map.max()

        fig, ax = plt.subplots(1, 5, figsize=(35//2, 28//2))
        ax[0].imshow(im.permute(1, 2, 0))
        ax[1].imshow(to_show.permute(1, 2, 0), cmap='gray')
        ax[2].imshow(empty_img.permute(1, 2, 0), cmap='gray')
        ax[3].imshow(attribution_map, vmin=0, vmax=1,cmap='hot')

        numer = attribution_map - attribution_map.min()
        denom = (attribution_map.max() - attribution_map.min()) + 1e-5
        attribution_map = numer / denom

        # th, _ = attribution_map.reshape(-1).sort()
        # th = th[-int((len(th)*.01))]
        # attribution_map[attribution_map >= th] = 1.
        # attribution_map[attribution_map < th] = 0.
        ax[4].imshow(attribution_map,
                        vmin=attribution_map.min(),
                        vmax=attribution_map.max(),
                        cmap='gray')

        # ax[2].imshow(empty_img, cmap='gray')

        if d_s[1]['class'][0] == 'Normal':
            s_title = ' ' + 'Ground Truth: No cancer'  # + sid
        elif d_s[1]['class'][0] == 'Benign':
            s_title = ' ' + 'Ground Truth: No cancer'  # + sid
        elif d_s[1]['class'][0] == 'Malignant':
            s_title = ' ' + 'Ground Truth: Malignant'  # + sid
        elif d_s[1]['class'][0] == 'Lymph_nodes':
            s_title = ' ' + 'Ground Truth: Lymph_nodes'  # + sid
        ax[0].set_title(s_title)
        ax[1].set_title("Selected tiles")
        # predictions = ['No cancer', ' Cancer']
        predictions = class_names
        # bce / ce
        pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
        s_title = "Predicted: " + predictions[int(pred[0])] +\
            '[' + str(score[0][0].__round__(2)) + ']'
        # [0][0] clam ds // [0][0][0] ag
        ax[2].set_title(s_title)
        ax[3].set_title("GradCAM")
# %%