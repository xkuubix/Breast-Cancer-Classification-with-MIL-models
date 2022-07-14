from math import floor
import torch
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset, DataLoader


def data_loader_gen(datasets: list,
                    test_valid_ratio: float,
                    valid_test_ratio: float,
                    batch_size: int) -> tuple:
    '''
    Generate equally distributed classes among dataloaders
    for given ratios.
    Returns tuple of dataloaders (train | validation | test)
    '''

    ds_len = len(datasets)
    min_size = min([len(datasets[item]) for item in range(ds_len)])
    used_seed = torch.manual_seed(42)

    train_count = floor(test_valid_ratio*min_size)
    val_count = floor((min_size - train_count)*valid_test_ratio)
    test_count = floor((min_size - train_count)*(1-valid_test_ratio))

    # add remaining samples to train set
    if train_count + val_count + test_count < min_size:
        train_count = min_size - val_count - test_count

    # create subsets from datasets with len=min_size
    subsets = [Subset(datasets[item],
                      range(0, min_size)) for item in range(ds_len)]

    counts = [train_count, val_count, test_count]

    n_tr, n_val, n_tst = torch.utils.data.random_split(subsets[0],
                                                       counts,
                                                       generator=used_seed)
    b_tr, b_val, b_tst = torch.utils.data.random_split(subsets[1],
                                                       counts,
                                                       generator=used_seed)
    m_tr, m_val, m_tst = torch.utils.data.random_split(subsets[2],
                                                       counts,
                                                       generator=used_seed)
    l_tr, l_val, l_tst = torch.utils.data.random_split(subsets[3],
                                                       counts,
                                                       generator=used_seed)

    train_set = ConcatDataset([n_tr, b_tr, m_tr, l_tr])
    val_set = ConcatDataset([n_val, b_val, m_val, l_val])
    test_set = ConcatDataset([n_tst, b_tst, m_tst, l_tst])

    print("Train set:", len(train_set), "images",
          "\nTest  set:", "", len(test_set), "images",
          "\nValid set:", "", len(val_set), "images")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=1,
                              pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=True, num_workers=1,
                             pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=True, num_workers=1,
                            pin_memory=True)
    ds = [ds_len*item for item in counts]

    data_loaders_sizes = {"train": ds[0], "val": ds[1], "test": ds[2]}
    data_loaders = {"train": train_loader,
                    "val": val_loader,
                    "test": test_loader}

    return data_loaders, data_loaders_sizes
