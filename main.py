# %%
import torch
from torchvision import transforms as T
from my_drawings import my_show_image, my_show_training_results
from my_stats import make_df  # , show_stats
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
transform = T.Compose([  # T.RandomAffine(degrees=(3), translate=(0, 0.1)),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                       ])
transforms_val_test = None
tfs = [transform, transforms_val_test, transforms_val_test]

# GET TILES (PATCHES) COORDS (size/overlap)-----------------------------------
tiles_train = get_tiles(3518, 2800,
                        patch_size, patch_size,
                        overlap_train)
tiles_test_val = get_tiles(3518, 2800,
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
# TRAIN NETWORK---------------------------------------------------------------
run = neptune.init(project='jakub-buler/BCC')
run['config'] = config

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
if 1:
    net_BACC = net
    net_BACC.load_state_dict(state_dict_BACC)
    metrics, cm = test_net(net_BACC, data_loaders, class_names,
                           device, drawing_num=0)
    unique_filename1 = str(uuid.uuid4())
    model_save_path = ('/media/dysk/student2/mammografia/Zapisy/'
                       + 'neptune_saved_models/' + unique_filename1)

    run['test/BACC/metrics'].log(metrics)
    run['test/BACC/conf_mx'].upload(cm)
    run['test/BACC/file_name'] = unique_filename1

    torch.save(net_BACC.state_dict(), model_save_path)
    del net_BACC

if 1:
    net_BL = net
    net_BL.load_state_dict(state_dict_BL)
    metrics, cm = test_net(net_BL, data_loaders, class_names,
                           device, drawing_num=0)
    unique_filename2 = str(uuid.uuid4())
    model_save_path = ('/media/dysk/student2/mammografia/Zapisy/'
                       + 'neptune_saved_models/' + unique_filename2)

    run['test/BL/metrics'].log(metrics)
    run['test/BL/conf_mx'].upload(cm)
    run['test/BL/file_name'] = unique_filename2

    torch.save(net_BL.state_dict(), model_save_path)
    del net_BL

run.stop()
# %%
if 0:
    fn = '54650285-9f0d-4ff8-bb5e-fd2fa05d3904'
    model_load_path = (
        '/media/dysk/student2/mammografia/Zapisy/'
        + 'neptune_saved_models/' + fn)
    net.load_state_dict(torch.load(model_load_path))
# %%
if 0:
    with torch.no_grad():
        for i in range(1):
            empty_img = torch.zeros(3, 3518, 2800)
            to_show = torch.zeros(3, 3518, 2800)

            ds = next(iter(data_loaders['test']))
            im = ds[1]['full_image'].squeeze(dim=0)
            id = ds[1]['tiles_indices'].squeeze(dim=0)

            net.eval()
            input = ds[0].to(device)
            output, weights = net(input)
            weights = weights.reshape(-1)
            for item in range(len(id)):
                h_min, w_min, dh, dw = tiles[1][id][item]
                to_show[:, h_min:h_min+dh, w_min:w_min+dw] = 1
                empty_img[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                    weights[item].detach().cpu()

            empty_img /= torch.max(empty_img.reshape(-1))
            # empty_img *= im

            fig, ax = plt.subplots(1, 3, figsize=(35//2, 28//2))
            ax[0].imshow(im.permute(1, 2, 0), cmap='gray')
            ax[1].imshow(to_show.permute(1, 2, 0), cmap='gray')
            ax[2].imshow(empty_img.permute(1, 2, 0), cmap='gray')
            if ds[1]['labels'] == 0.:
                s_title = ' ' + 'Ground Truth: No Cancer'
            elif ds[1]['labels'] == 1.:
                s_title = ' ' + 'Ground Truth: Cancer'
            ax[0].set_title(s_title)
            ax[1].set_title("Selected tiles")
            predictions = ['No cancer', ' Cancer']
            pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
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
