# %%
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

import matplotlib.pyplot as plt  # temp
# from tile import convert_img_to_bag, get_tiles  # temp

import argparse
import yaml
import neptune.new as neptune
import uuid
import copy


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

patch_size = config['data_sets']['patch_size']
overlap_train = config['data_sets']['overlap_train_val']
overlap_val_test = config['data_sets']['overlap_val_test']

# TRANSFORMS------------------------------------------------------------------
cj_prob = 0.5
cj_bright = 0.25
cj_contrast = 0.25
cj_sat = 0.25
cj_hue = 0.
gaussian_blur_prob = 0.5

input_size = patch_size
min_scale = 0.8

color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)
gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

transform = T.Compose([T.RandomAffine(degrees=(90), translate=(0, 0.1)),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       T.RandomResizedCrop(size=input_size,
                                           scale=(min_scale, 1.0)),
                       # T.RandomApply([color_jitter], p=cj_prob),
                       # T.RandomApply([gaussian_blur], p=gaussian_blur_prob)
                       ])

transforms_val_test = None
tfs = [transform, transforms_val_test, transforms_val_test]

# GET TILES (PATCHES) COORDS (size/overlap)-----------------------------------
tiles_train = get_tiles(3518*2, 2800,
                        patch_size, patch_size,
                        overlap_train)
tiles_test_val = get_tiles(3518*2, 2800,
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
        tiles=tiles
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
    import os
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
            # net.momentum = 0.01

    net.apply(deactivate_batchnorm)
if 1:
    # fn = 'clr_lr-1loss2.578792095184326'
    fn = 'clr_lr-0.001loss2.6703104972839355'
    model_load_path = ('/media/dysk/student2/mammografia/Zapisy/'
                       + 'clr_saved_models/' + fn)
    sd = torch.load(model_load_path, map_location=device)
    net_sd = copy.deepcopy(net.state_dict())
    # fe = 'feature_extractor'
    fe = 'backbone'

    net_fe_layers = list()
    for name, param in net_sd.items():
        if 'feature_extractor' in name:
            net_fe_layers.append(name)

    for i, (name, param) in enumerate(sd.items()):
        if fe in name:
            param = param.data
            net_sd[net_fe_layers[i]].copy_(param)
    net.load_state_dict(net_sd)

    # ct = 0
    # gct = 0
    # for child in net.children():
    #     ct += 1
    #     if ct == 1:
    #         for grandchild in child.children():
    #             gct += 1
    #             if gct > 7:
    #                 for param in grandchild.parameters():
    #                     param.requires_grad = False

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
    neptune_run=run)

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
if 0:
    std = torch.load('/media/dysk/student2/mammografia/Zapisy/'
                     + 'neptune_saved_models/'
                     + 'd855fb28-53c8-416f-9fa6-099f3b5a6b2b',
                     map_location=device)
    net.load_state_dict(std)
# %%
if 0:
    with torch.no_grad():
        for i in range(1):
            empty_img = torch.zeros(3, 3518, 2800)
            to_show = torch.zeros(3, 3518, 2800)

            d_s = next(iter(data_loaders['test']))
            im = d_s[1]['full_image'].squeeze(dim=0)
            id = d_s[1]['tiles_indices'].squeeze(dim=0)

            net.eval()
            input = d_s[0].to(device)
            output = net(input, d_s[1]['tile_cords'])
            pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
            # _, pred = torch.max(output[0].reshape(-1, 4), 1)
            # if pred != d_s[1]['labels'] or d_s[1]['labels'] == 0:
            #     continue
            weights = output[1]

            # weights = torch.sum(output[2], dim=1)
            plt.matshow(output[2].cpu().numpy())
            weights = weights.squeeze()

            for item in range(len(id)):
                h_min, w_min, dh, dw, _, _ = tiles[1][id][item]
                to_show[:, h_min:h_min+dh, w_min:w_min+dw] = 1

                # if ind[item] == 0:
                #     empty_img[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                #         weights[item][pred].detach().cpu()

                empty_img[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                    weights[item].detach().cpu()

                # empty_img[:, h_min:h_min+dh, w_min:w_min+dw] =\
                #     torch.max(empty_img[:, h_min:h_min+dh, w_min:w_min+dw],
                #               weights[item].detach().cpu())
            empty_img -= torch.min(weights.reshape(-1).cpu())
            empty_img[empty_img < 0] = 0
            empty_img /= torch.max(empty_img.reshape(-1))
            # empty_img[empty_img < 0.9] = 0
            # empty_img /= torch.max(empty_img.reshape(-1))
            # empty_img *= 100

            fig, ax = plt.subplots(1, 3, figsize=(35//2, 28//2))
            ax[0].imshow(im.permute(1, 2, 0), cmap='gray')
            ax[1].imshow(to_show.permute(1, 2, 0), cmap='gray')
            ax[2].imshow(empty_img.permute(1, 2, 0), cmap='gray')

            if d_s[1]['class'][0] == 'Normal':
                s_title = ' ' + 'Ground Truth: Normal'
            elif d_s[1]['class'][0] == 'Benign':
                s_title = ' ' + 'Ground Truth: Benign'
            elif d_s[1]['class'][0] == 'Malignant':
                s_title = ' ' + 'Ground Truth: Malignant'
            elif d_s[1]['class'][0] == 'Lymph_nodes':
                s_title = ' ' + 'Ground Truth: Lymph_nodes'
            ax[0].set_title(s_title)
            ax[1].set_title("Selected tiles")
            # predictions = ['No cancer', ' Cancer']
            predictions = class_names
            # bce / ce
            # pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
            s_title = "Predicted: " + predictions[int(pred[0])]
            ax[2].set_title(s_title)
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
