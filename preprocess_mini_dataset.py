from operator import delitem
import os
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import json

import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

levelsetlayer = LevelSetLayer2D(size=(85, 85), maxdim=1, sublevel=False)

dir = 'images/different_thresholds'
original_dir = dir + '/original'
core_dir = dir + '/core'
neighborhood_dir = dir + '/neighborhood'
diff_dir = dir + '/diff'
processed_dir = dir + '/processed'

betas = []

barcode_dict = {}

otsu_thresholds = np.loadtxt('data/mini_otsu_thresholds_diff_thresholds.csv', delimiter=',')
epsilon = 15

def get_betas(img_array):
    img = torch.Tensor(img_array) / 255
    barcode = levelsetlayer(img)[0]
    beta0 = 0
    for pair in barcode[0]:
        length = pair[0] - pair[1]
        if length >= 1.0:
            beta0 += 1
    beta1 = 0
    for pair in barcode[1]:
        length = pair[0] - pair[1]
        if length >= 1.0:
            beta1 += 1 
    ans = [beta0, beta1]
    return ans   

n = len(os.listdir(original_dir))
core_betas = []
ngh_betas = []
for i in range(n):

    filename = 'data_' + str(i) + '.png'
    img_original = Image.open(os.path.join(original_dir, filename))
    img = np.asarray(img_original)

    N = otsu_thresholds[i] - 15
    C = otsu_thresholds[i] + 15

    neighborhood = img.copy()
    neighborhood[neighborhood >= N] = 255
    neighborhood[neighborhood < N] = 0
    ngh_betas.append(get_betas(neighborhood))
    Image.fromarray(neighborhood).save(os.path.join(neighborhood_dir, filename))

    core = img.copy()
    core[core >= C] = 255
    core[core < C] = 0
    core_betas.append(get_betas(core))
    Image.fromarray(core).save(os.path.join(core_dir, filename))

    diff = img.copy()
    diff[neighborhood == 0] = 0 # set pixels outside of neighborhood and inside core to 0
    diff[core == 255] = 0
    Image.fromarray(diff).save(os.path.join(diff_dir, filename))

    # make images with 0, 1/2, 1 values
    new_img = img.copy()
    processed = np.zeros(img.shape)
    processed[new_img >= N] = 0.5
    processed[new_img >= C] = 1
    
    # calculate target beta numbers, save barcodes of preprocessed images.
    row = np.zeros(2)
    barcode = levelsetlayer(torch.Tensor(processed))[0]
    barcode0 = barcode[0]
    goal_b0 = 0
    for bar in barcode0:
        if bar[0] == 1 and bar[1] <= 0:
            goal_b0 += 1
    row[0] = goal_b0

    barcode1 = barcode[1]
    goal_b1 = 0
    for bar in barcode1:
        if bar[0] == 1 and bar[1] <= 0:
            goal_b1 += 1
    row[1] = goal_b1

    np.savetxt('data/barcodes/preprocessed_mini_dataset_different_thresholds/dim0/preprocessed_barcode_dim_0_data_%d.csv' % i, np.asarray(barcode0), fmt='%1.1f', delimiter=',')
    np.savetxt('data/barcodes/preprocessed_mini_dataset_different_thresholds/dim1/preprocessed_barcode_dim_1_data_%d.csv' % i, np.asarray(barcode1), fmt='%1.1f', delimiter=',')

    betas.append(row)
    print(str(i), ' ', row)
    processed *= 255
    processed = processed.astype(np.uint8)

    Image.fromarray(processed).save(os.path.join(processed_dir, filename))
    
    i += 1

betas = np.stack(betas, axis=0)
np.savetxt('data/mini_betas_diff_thresholds.csv', betas, fmt='%1.0f', delimiter=',')

ngh_betas = np.stack(ngh_betas, axis=0)
np.savetxt('data/ngh_mini_betas.csv', ngh_betas, fmt='%1.0f', delimiter=',')

core_betas = np.stack(core_betas, axis=0)
np.savetxt('data/core_mini_betas.csv', core_betas, fmt='%1.0f', delimiter=',')