# %%
import torch
from my_stats import make_df
from choose_NCOS import choose_NCOS
from net_utils import test_net
from tile_maker import get_tiles
from BreastCancerDataset import CMMD_DS
from torch.utils.data import DataLoader
# import cv2
import matplotlib.pyplot as plt  # temp
# from tile import convert_img_to_bag, get_tiles  # temp
import argparse
import yaml
import os
import pandas as pd


# MAKE PARSER AND LOAD PARAMS FROM CONFIG FILE--------------------------------
def get_args_parser():
    default = '/home/student2/Projekt_MMG/config.yml'
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
             from_file=True,
             save_to_file=False,
             file_dir=file_dir)
if create_new_dl:
    os.chdir('/media/dysk/student2/CMMD/TheChineseMammographyDatabase/')
    test_set = pd.read_csv('CMMD_clinicaldata_revision_CSV.csv', sep=';')
    root = '/media/dysk/student2/CMMD/TheChineseMammographyDatabase/CMMD/'
    data_loaders = {}
    dl_sizes = {}
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


#     data_loaders, data_loaders_sizes, ds = gen_data_loader(
#         root=root,
#         train_df=train_set,
#         val_df=val_set,
#         test_df=test_set,
#         view=view,
#         transforms=tfs,
#         batch_size=batch_size,
#         nw=num_workers,
#         conv_to_bag=True,
#         bag_size_train=bag_size_train,
#         bag_size_val_test=bag_size_val_test,
#         tiles=tiles,
#         img_size=image_size,
#         is_multimodal=image_multimodality
#         )
#     torch.save(data_loaders, dataloader_path)
#     torch.save(data_loaders_sizes, dataloader_sizes_path)
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
criterion_type = config['training_plan']['criterion']
optimizer_type = config['training_plan']['optim_name']
scheduler_type = config['training_plan']['scheduler']

if scheduler_type == 'one-cycle':
    scheduler_type['epochs'] = EPOCHS
    scheduler_type['steps_per_epoch'] = len(data_loaders['train'])

net, criterion, optimizer, scheduler = choose_NCOS(
    net_ar=net_ar,
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

    net.apply(deactivate_batchnorm)
if 1:
    std = torch.load('/media/dysk/student2/mammografia/Zapisy/'
                     + 'neptune_saved_models/'
                     + '49f86433-a8df-42be-97fb-d8f71127f120',
                     map_location=device)
    net.load_state_dict(std)
# %%
out = test_net(net, data_loaders, class_names, device)
# %%
if 1:
    with torch.no_grad():
        for i in range(5):
            empty_img = torch.zeros(3, image_size[0], image_size[1])
            empty_img_counts = torch.ones(3, image_size[0], image_size[1])
            to_show = torch.ones(3, image_size[0], image_size[1])

            d_s = next(iter(data_loaders['test']))
            im = d_s[1]['full_image'].squeeze(dim=0)
            id = d_s[1]['tiles_indices'].squeeze(dim=0)
            net.eval()
            input = d_s[0].to(device)
            output = net(input, None)  # , d_s[1]['tile_cords'])
            score = torch.sigmoid(output[0]).detach().cpu().numpy()
            pred = score.round()
            # _, pred = torch.max(output[0].reshape(-1, 4), 1)
            # if pred != d_s[1]['labels'] or d_s[1]['labels'] == 0:
            #     continue

            # weights = output[1]
            weights = output[3]
            weights = weights.squeeze()

            for item in range(len(id)):
                h_min, w_min, dh, dw, _, _ = tiles[1][id][item]
                to_show[:, h_min:h_min+dh, w_min:w_min+dw] = d_s[0][0][item]

                empty_img[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                    weights[item].detach().cpu()
                empty_img_counts[:, h_min:h_min+dh, w_min:w_min+dw] += 1

            # empty_img = torch.div(empty_img*2, empty_img_counts)
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
                s_title = ' ' + 'Ground Truth: Cancer'  # + sid
            elif d_s[1]['class'][0] == 'Lymph_nodes':
                s_title = ' ' + 'Ground Truth: Cancer'  # + sid
            ax[0].set_title(s_title)
            ax[1].set_title("Selected tiles")
            # predictions = ['No cancer', ' Cancer']
            predictions = class_names
            # bce / ce
            pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
            s_title = "Predicted: " + predictions[int(pred[0])] +\
                '[' + str(score[0][0].__round__(2)) + ']'
            ax[2].set_title(s_title)


# %%
