import torch
import numpy as np


def get_tiles(hTile=224, wTile=224):

    '''
    Returns tiles, default tile size: 224px 224px
    Number of tiles depends on image size and
    tile parameters
    '''
    image = torch.empty(3, 3500, 2600)
    h, w = image.shape[1:]
    # print(h, w)
    # Number of tiles
    nTilesX = np.uint8(np.ceil(w / wTile))
    nTilesY = np.uint8(np.ceil(h / hTile))
    # print('x/y tiles', nTilesX, nTilesY)

    # Total remainders
    remainderX = nTilesX * wTile - w
    remainderY = nTilesY * hTile - h

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(
                                                    remainderX / (nTilesX-1)))
    remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(
                                                    remainderY / (nTilesY-1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1

    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

    # Determine proper tile boxes
    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if (j < (nTilesY-1)):
                y = y + hTile - remaindersY[j]
        if (i < (nTilesX-1)):
            x = x + wTile - remaindersX[i]
    return tiles


def convert_img_to_bag(image, tiles, bag_size):
    hTile = tiles[0][3]
    wTile = tiles[0][2]
    c = image.shape[0]
    img_shape = (len(tiles), c, hTile, wTile)
    new_img = torch.empty(img_shape)
    px_sum = np.empty(len(tiles))
    for i, tile in enumerate(tiles):
        for channel in range(c):
            new_img[i][channel] = image[channel][
                       tile[1]:tile[1]+tile[3], tile[0]:tile[0]+tile[2]]
        px_sum[i] = new_img[i].reshape(1, -1).sum(-1)/c

    sorted_tiles_idx = np.argsort(-px_sum)
    # print(sorted_tiles_idx)
    # print(px_sum)
    # za_1k = (px_sum > 1000).sum()
    # za_10k = (px_sum > 10000).sum()
    # print(za)
    if bag_size > len(sorted_tiles_idx):
        bag_size = len(sorted_tiles_idx)
    bag = new_img[sorted_tiles_idx[:bag_size]]
    return bag
