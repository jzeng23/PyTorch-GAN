# code from https://en.wikipedia.org/wiki/Otsu%27s_method

import numpy as np
from PIL import Image
import os

def compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1

def generate_csv(dir, save_path):
    thresholds = np.zeros(len(os.listdir(dir)))
    for i in range(len(os.listdir(dir))):
        filename = 'data_' + str(i) + '.png'
        im = Image.open(os.path.join(dir, filename))
        im = np.asarray(im)
        threshold_range = range(np.max(im)+1)
        criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
        best_threshold = threshold_range[np.argmin(criterias)]
        thresholds[i] = best_threshold
        print(i)
    np.savetxt(save_path, thresholds, delimiter=',')

img_dir='images/full_dataset_different_thresholds'
train_dir = img_dir + '/train/original'
test_dir = img_dir + '/test/original'
generate_csv(train_dir, 'data/otsu_thresholds_full_train.csv')

# testing all thresholds from 0 to the maximum of the image


# best threshold is the one minimizing the Otsu criteria
