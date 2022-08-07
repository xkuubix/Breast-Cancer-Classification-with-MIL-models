import numpy as np
import matplotlib.pyplot as plt


def get_points(ds, show_images_count: int):
    r_up = np.array([])
    r_down = np.array([])
    c = np.array([])
    i = 0
    for d_s in ['train', 'val', 'test']:
        for image, target in ds[d_s]:
            my_mask = image[0].detach()
            my_mask[my_mask < 1e-18] = 0
            my_mask[my_mask > 1e-18] = 1
            np_im = my_mask.numpy()
            rows = np.argwhere(np.sum(np_im, axis=1) > 10)
            cols = np.argwhere(np.sum(np_im, axis=0) > 10)
            col1 = np.argmax(cols)
            row1 = rows[0]
            row2 = rows[-1]
            if col1 < 2700:
                r_up = np.append(r_up, row1)
                r_down = np.append(r_down, row2)
                c = np.append(c, col1)
            if i < show_images_count:
                if col1 >= 2700:
                    # new_image = np_im[row1[0]-50:row2[0]+50, 0:col1+50]
                    print(target['file'])
                    new_image = np_im[:, 0:2590]
                    print(col1)
                    plt.imshow(new_image)
                    plt.show()
                    # plt.imshow(image[0])
                    # plt.title(str(i))
                    # plt.show()
                    i += 1
    print(np.min(r_up), np.max(r_down), np.max(c))
    return np.min(r_up), np.max(r_down), np.max(c)
