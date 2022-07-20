import os
import numpy as np
from PIL import Image
from numpy import asarray

import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

otsu_thresholds = np.loadtxt('data/mini_otsu_thresholds_diff_thresholds.csv', delimiter=',')
epsilon = 15
epoch = 8000
barcodes_dir = 'implementations/aae/barcode'
n = 1

betas = np.zeros((n,2))
for i in range(n):
    N = (otsu_thresholds[i] - epsilon) / 255
    C = (otsu_thresholds[i] + epsilon) / 255
    barcode0 = np.loadtxt(os.path.join(barcodes_dir, 'dim0/dim_0_epoch_%d.csv' % (epoch)), delimiter=',')
    beta0 = 0
    for k in range(barcode0.shape[0]):
        pair = barcode0[k, :]
        if pair[0] >= C and pair[1] < N:
            beta0 += 1
    betas[i, 0] = beta0

    barcode1 = np.loadtxt(os.path.join(barcodes_dir, 'dim1/dim_1_epoch_%d.csv' % (epoch)), delimiter=',')
    beta1 = 0
    for k in range(barcode1.shape[0]):
        pair = barcode1[k, :]
        if pair[0] >= C and pair[1] < N:
            beta1 += 1
    betas[i, 1] = beta1
np.savetxt('data/original_output_betas_epoch_%d.csv' % epoch, betas, fmt='%1.0f', delimiter=',')

