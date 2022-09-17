import torch
from torch import nn
import torchvision
from torchvision import transforms as T
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from my_stats import make_df  # , show_stats
from torch.optim import lr_scheduler
# %%
from gen_data_loader import gen_data_loader
from random_split_df import random_split_df
from tile_maker import get_tiles
import argparse
import yaml
import copy
import neptune.new as neptune


def deactivate_batchnorm(net):
    if isinstance(net, nn.BatchNorm2d):
        net.track_running_stats = False
        net.running_mean = None
        net.running_var = None

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

EPOCHS = config['training_plan']['parameters']['epochs'][0]
lr = config['training_plan']['parameters']['lr'][0]
wd = config['training_plan']['parameters']['wd'][0]
net_ar = config['training_plan']['architectures']['names']
criterion_type = config['training_plan']['criterion']
optimizer_type = config['training_plan']['optim_name']
scheduler_type = config['training_plan']['scheduler']

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
# %% TRANSFORMS
input_size = 224
cj_prob = 0.5
cj_bright = 0.2
cj_contrast = 0.2
cj_sat = 0.2
cj_hue = 0.2
min_scale = 0.8
gaussian_blur_prob = 0.5


color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)
gaussian_blur = T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))

clr_transform = [T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
                 T.RandomApply([color_jitter], p=cj_prob),
                 T.RandomApply([gaussian_blur], p=gaussian_blur_prob),
                 ]
aug = T.Compose(clr_transform)
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
        tiles=tiles,
        )
    torch.save(data_loaders, dataloader_path)
    torch.save(data_loaders_sizes, dataloader_sizes_path)
else:
    data_loaders = torch.load(dataloader_path)
    data_loaders_sizes = torch.load(dataloader_sizes_path)


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(512, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x.contiguous())
        return z


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = SimCLR(backbone)
model.apply(deactivate_batchnorm)

model.to(device)

criterion = NTXentLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler_type = config['training_plan']['scheduler']
step_szie = scheduler_type['step_size']
gamma = scheduler_type['gamma']
scheduler = lr_scheduler.StepLR(optimizer,
                                step_size=step_szie,
                                gamma=gamma)

best_loss = None

run = neptune.init(project='ProjektMMG/CLR')
run['config'] = config

print("Starting Training")
for epoch in range(EPOCHS):
    total_loss_train = 0
    total_loss_val = 0

    for x, _ in data_loaders['train']:
        model.train()
        x1 = aug(x.squeeze(0)).contiguous().to(device)
        x2 = aug(x.squeeze(0)).contiguous().to(device)
        z0 = model(x1)
        z1 = model(x2)
        loss = criterion(z0, z1)
        total_loss_train += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    avg_loss_train = total_loss_train / data_loaders_sizes['train']
    if run is not None:
        # NEPTUNE LOGGING
        run['train' + '/loss'].log(avg_loss_train)
    else:
        print(f"epoch: {epoch:>02}, train loss: {avg_loss_train:.5f}", end=' ')

    for x, _ in data_loaders['val']:
        with torch.no_grad():
            model.eval()
            x1 = aug(x.squeeze(0)).contiguous().to(device)
            x2 = aug(x.squeeze(0)).contiguous().to(device)
            z0 = model(x1)
            z1 = model(x2)
            loss = criterion(z0, z1)
            total_loss_val += loss.detach()
    avg_loss_val = total_loss_val / data_loaders_sizes['val']

    if best_loss is None or avg_loss_val < best_loss:
        best_loss = avg_loss_val
        best_net_wts = copy.deepcopy(model.state_dict())

    if run is not None:
        # NEPTUNE LOGGING
        run['val' + '/loss'].log(avg_loss_val)
    else:
        print(f"val loss: {avg_loss_val:.5f}")

filename = 'clr_lr-' + str(lr) + 'loss' + str(best_loss.item())
model_save_path = ('/media/dysk/student2/mammografia/Zapisy/'
                   + 'clr_saved_models/' + filename)
print('filename: ', filename)
print('best loss: ', best_loss)
torch.save(model.state_dict(), model_save_path)

if run is not None:
    run['clr/file_name'] = filename

if run is not None:
    run.stop()
