from BreastCancerDataset import BreastCancerDataset
# from BreastCancerDataset import CMMD_DS
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import numpy as np


def gen_data_loader(root,
                    train_df,
                    val_df,
                    test_df,
                    view: list,
                    transforms: list,
                    batch_size=2,
                    nw=4,
                    conv_to_bag=False,
                    bag_size_train=100,
                    bag_size_val_test=100,
                    tiles=None,
                    collate_fn=None,
                    img_size=[3518, 2800],
                    is_multimodal=False
                    ) -> tuple:

    train_dataset = BreastCancerDataset(root, train_df,
                                        view, transforms[0],
                                        conv_to_bag,
                                        bag_size=bag_size_train,
                                        tiles=tiles[0],
                                        img_size=img_size,
                                        is_multimodal=is_multimodal
                                        )
    val_dataset = BreastCancerDataset(root, val_df,
                                      view, transforms[1],
                                      conv_to_bag,
                                      bag_size=bag_size_val_test,
                                      tiles=tiles[1],
                                      img_size=img_size,
                                      is_multimodal=is_multimodal
                                      )
    test_dataset = BreastCancerDataset(root, test_df,
                                       view, transforms[2],
                                       conv_to_bag,
                                       bag_size=bag_size_val_test,
                                       tiles=tiles[1],
                                       img_size=img_size,
                                       is_multimodal=is_multimodal
                                       )

    print_ds_info(train_dataset, 'Train dataset ', view)
    print_ds_info(val_dataset, 'Validation dataset ', view)
    print_ds_info(test_dataset, 'Test dataset ', view)

    labels = []
    for i in range(len(train_dataset)):
        # print(train_dataset[i][1]['labels'].item())
        labels.append(train_dataset[i][1]['labels'].item())

    class_sample_count = torch.tensor(
        [(torch.from_numpy(np.array(labels)) == t).sum() for t in torch.unique(
            torch.from_numpy(np.array(labels)), sorted=True)])
    # print(class_sample_count)
    weight = 1. / class_sample_count.float()
    # print(weight)
    samples_weight = torch.tensor(
        [weight[int(t)] for t in torch.from_numpy(np.array(labels))])
    # print(samples_weight)
    sampler = WeightedRandomSampler(samples_weight,
                                    num_samples=len(samples_weight),
                                    # num_samples=class_sample_count[1].item()*2,
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              #  shuffle=True,
                              num_workers=nw,
                              collate_fn=collate_fn,
                            #   sampler=sampler,
                              pin_memory=True
                              )
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=True, num_workers=nw,
                            collate_fn=collate_fn,
                              pin_memory=True
                            )

    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=True, num_workers=nw,
                             collate_fn=collate_fn
                             )

    dl_sizes = [len(item) for item in [train_loader,
                                       val_loader,
                                       test_loader]]

    data_loaders_sizes = {"train": dl_sizes[0],
                          "val": dl_sizes[1],
                          "test": dl_sizes[2]}

    data_loaders = {"train": train_loader,
                    "val": val_loader,
                    "test": test_loader}

    data_sets = {"train": train_dataset,
                 "val": val_dataset,
                 "test": test_dataset}

    return data_loaders, data_loaders_sizes, data_sets


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
