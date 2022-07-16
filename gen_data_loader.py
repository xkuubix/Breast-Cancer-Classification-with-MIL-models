from BreastCancerDataset import BreastCancerDataset
from torch.utils.data import DataLoader


def data_loader_gen(root,
                    train_df,
                    val_df,
                    test_df,
                    view,
                    transforms=None,
                    batch_size=2
                    ) -> tuple:

    train_dataset = BreastCancerDataset(root, train_df,
                                        view, transforms)
    val_dataset = BreastCancerDataset(root, val_df,
                                      view, transforms)
    test_dataset = BreastCancerDataset(root, test_df,
                                       view, transforms)

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

    return data_loaders, data_loaders_sizes
