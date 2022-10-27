import torch
import numpy as np
from sklearn.utils import shuffle


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points


def get_tiles(h, w,
              hTile=224,
              wTile=224,
              overlap=0.5):

    X_points = start_points(w, wTile, overlap)
    Y_points = start_points(h, hTile, overlap)

    tiles = np.zeros((len(Y_points)*len(X_points), 6), np.int)

    k = 0
    for i, y_cord in enumerate(Y_points):
        for j, x_cord in enumerate(X_points):
            # split = image[i:i+hTile, j:j+wTile]
            tiles[k] = (y_cord, x_cord, hTile, wTile, i, j)
            k += 1
    return tiles


def convert_img_to_bag(image, tiles, bag_size):
    hTile = tiles[0][2]
    wTile = tiles[0][3]
    c = image.shape[0]
    img_shape = (len(tiles), c, hTile, wTile)
    new_img = torch.zeros(img_shape)
    px_sum = torch.zeros(len(tiles))
    px_non_zero = torch.zeros(len(tiles), dtype=torch.float32)

    for i, tile in enumerate(tiles):
        for channel in range(c):
            new_img[i][channel] = image[channel][tile[0]:tile[0]+tile[2],
                                                 tile[1]:tile[1]+tile[3]]
        px_sum[i] = new_img[i].reshape(1, -1).sum(-1) / c
        px_non_zero[i] = (new_img[i][0].reshape(1, -1) > 0).sum() \
            / len(new_img[i][0].reshape(1, -1)) / (hTile * wTile) * 100

    sorted_tiles_idx = np.argsort(-px_sum)

    px_non_zero_75pc = (px_non_zero > 75).sum()

    # if bag_size > len(sorted_tiles_idx) and not None:
    #     bag_size = len(sorted_tiles_idx)
    # # bag = new_img[sorted_tiles_idx[:bag_size]]
    if bag_size == 'const50':
        instances = new_img[sorted_tiles_idx[:50]]
        instances_idx = sorted_tiles_idx[:50]
        instances_cords = tiles[instances_idx, 4:6]
        instances, instances_idx, instances_cords = shuffle(
            instances, instances_idx, instances_cords)
    elif bag_size == 'const150':
        instances = new_img[sorted_tiles_idx[:100]]
        instances_idx = sorted_tiles_idx[:100]
        instances_cords = tiles[instances_idx, 4:6]
        instances, instances_idx, instances_cords = shuffle(
            instances, instances_idx, instances_cords)
    elif bag_size == 'const300':
        instances = new_img[sorted_tiles_idx[:300]]
        instances_idx = sorted_tiles_idx[:300]
        instances_cords = tiles[instances_idx, 4:6]
        instances, instances_idx, instances_cords = shuffle(
            instances, instances_idx, instances_cords)
    elif bag_size is not -1:
        instances = new_img[sorted_tiles_idx[:px_non_zero_75pc]]
        instances_idx = sorted_tiles_idx[:px_non_zero_75pc]
        instances_cords = tiles[instances_idx, 4:6]
        instances, instances_idx, instances_cords = shuffle(
            instances, instances_idx, instances_cords)
        if px_non_zero_75pc > bag_size:
            instances = instances[:bag_size]
            instances_idx = instances_idx[:bag_size]
            instances_cords = tiles[instances_idx, 4:6]
    elif bag_size == -1:
        instances = new_img[sorted_tiles_idx[:px_non_zero_75pc]]
        instances_idx = sorted_tiles_idx[:px_non_zero_75pc]
        instances_cords = tiles[instances_idx, 4:6]
        instances, instances_idx, instances_cords = shuffle(
            instances, instances_idx, instances_cords)
    # elif bag_size < px_non_zero_75pc:
    #     instances = new_img[sorted_tiles_idx[:px_non_zero_75pc]]
    #     instances_idx = sorted_tiles_idx[:px_non_zero_75pc]
    #     instances, instances_idx = shuffle(instances, instances_idx)
    #     instances = instances[:bag_size]
    #     instances_idx = instances_idx[:bag_size]

    # else:
    #     instances = new_img[sorted_tiles_idx[:bag_size]]
    #     instances_idx = sorted_tiles_idx[:bag_size]
    #     instances, instances_idx = shuffle(instances, instances_idx)

    return instances, instances_idx, instances_cords
