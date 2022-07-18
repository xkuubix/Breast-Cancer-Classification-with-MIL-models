from BreastCancerDataset import BreastCancerDataset
from torch.utils.data import DataLoader


def gen_data_loader(root,
                    train_df,
                    val_df,
                    test_df,
                    view,
                    transforms: list,
                    batch_size=2
                    ) -> tuple:

    train_dataset = BreastCancerDataset(root, train_df,
                                        view, transforms[0])
    val_dataset = BreastCancerDataset(root, val_df,
                                      view, transforms[1])
    test_dataset = BreastCancerDataset(root, test_df,
                                       view, transforms[2])

    print_ds_info(train_dataset, 'Train dataset ', view)
    print_ds_info(val_dataset, 'Validation dataset ', view)
    print_ds_info(test_dataset, 'Test dataset ', view)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=1,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=1,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=1,
                             pin_memory=True)

    ds_sizes = [len(item) for item in [train_dataset,
                                       val_dataset,
                                       test_dataset]]

    data_loaders_sizes = {"train": ds_sizes[0],
                          "val": ds_sizes[1],
                          "test": ds_sizes[2]}

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
