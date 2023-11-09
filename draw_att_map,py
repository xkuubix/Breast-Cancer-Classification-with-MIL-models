# %%
import os
import pandas as pd
import numpy as np
import torch
from my_stats import make_df
from choose_NCOS import choose_NCOS
# from net_utils import train_net, test_net
from tile_maker import get_tiles
# import cv2
from torch.utils.data import DataLoader
from BreastCancerDataset import CMMD_DS
from BreastCancerDataset import BreastCancerDataset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt  # temp
# from tile import convert_img_to_bag, get_tiles  # temp
import argparse
import yaml


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

#####################################
net_to_load = 'ea8571f2-7de0-479c-86b9-261265fb2d52'
fold_to_load = 0
#####################################


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

net_ar_dropout = config['training_plan']['architectures']['dropout']

image_size = config['image']['size']
image_multimodality = config['image']['multimodal']

patch_size = config['data_sets']['patch_size']
overlap_train = config['data_sets']['overlap_train_val']
overlap_val_test = config['data_sets']['overlap_val_test']

tfs = [None, None, None]

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
data_loaders = {}
dl_sizes = {}

kf = KFold(n_splits=5, shuffle=True, random_state=seed)

for fold, (train_val_idx, test_idx) in enumerate(kf.split(df)):
    if fold < fold_to_load:
        continue
    print('------------fold no---------{}----------------------'.format(fold))
    # print('Test indexes: ', test_idx)

    train_val_df = df.drop(index=test_idx)
    test_df = df.drop(index=train_val_idx)
    train_df = train_val_df.sample(frac=0.875, random_state=seed)
    val_df = train_val_df.drop(train_df.index)
    # print(len(train_df), len(val_df), len(test_df))
    # print('Train')
    # print(train_df['class'].value_counts())
    # print()
    # print('Val')
    # print(val_df['class'].value_counts())

    # print()
    # print('Test')
    # print(test_df['class'].value_counts())
    test_dataset = BreastCancerDataset(root, test_df,
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
    break
# %%
# SET TRAINING PLAN-----------------------------------------------------------

lr = config['training_plan']['parameters']['lr'][0]
wd = config['training_plan']['parameters']['wd'][0]
net_ar = config['training_plan']['architectures']['names']
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
if 1:
    from torch import nn

    def deactivate_batchnorm(net):
        if isinstance(net, nn.BatchNorm2d):
            net.track_running_stats = False
            net.running_mean = None
            net.running_var = None

if 1:
    net.apply(deactivate_batchnorm)
    std = torch.load('/media/dysk_a/jr_buler/mammografia/Zapisy/'
                     + 'neptune_saved_models/'
                     + net_to_load,
                     map_location=device)
    net.load_state_dict(std)
# %%
if 1:
    with torch.no_grad():
        for i in range(1):

            empty_img = torch.zeros(3, image_size[0], image_size[1])
            empty_img_counts = torch.ones(3, image_size[0], image_size[1])
            to_show = torch.ones(3, image_size[0], image_size[1])

            d_s = next(iter(data_loaders['test']))

            im = d_s[1]['full_image'].squeeze(dim=0)
            id = d_s[1]['tiles_indices'].squeeze(dim=0)

            sid = '[' + d_s[1]['patient_id'][0] + ']'
            # if d_s[1]['patient_id'][0] not in ['276926']: #, '299973', '1202672']:
            #     continue
            # print(d_s[1]['patient_id'][0])

            net.eval()
            input = d_s[0].to(device)
            output = net(input, None)  # , d_s[1]['tile_cords'])
            score = torch.sigmoid(output[0]).detach().cpu().numpy()
            pred = score.round()
            # _, pred = torch.max(output[0].reshape(-1, 4), 1)
            # if pred != d_s[1]['labels'] or d_s[1]['labels'] == 0:
            #     continue

            # gmil
            # weights = output[1]

            # clam
            weights = output[3]
            weights = weights.squeeze()

            for item in range(len(id)):
                h_min, w_min, dh, dw, _, _ = tiles[1][id][item]
                to_show[:, h_min:h_min+dh, w_min:w_min+dw] = d_s[0][0][item]

                empty_img[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                    weights[item].detach().cpu()
                empty_img_counts[:, h_min:h_min+dh, w_min:w_min+dw] += 1

            empty_img = torch.div(empty_img, empty_img_counts)
            empty_img /= torch.max(empty_img.reshape(-1))

            fig, ax = plt.subplots(1, 3, figsize=(35//2, 28//2))
            ax[0].imshow(im.permute(1, 2, 0))
            ax[1].imshow(to_show.permute(1, 2, 0), cmap='gray')
            ax[2].imshow(empty_img.permute(1, 2, 0), cmap='gray')
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
# %%
