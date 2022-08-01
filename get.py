import os
import numpy as np
from PIL import Image
from numpy import asarray

import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

otsu_thresholds = np.loadtxt('data/mini_otsu_thresholds_diff_thresholds.csv', delimiter=',')
epsilon = 15
epoch = 6300
barcodes_dir = 'implementations/aae/barcode/loss_mse/lr_0.0002/alpha'
n = 10

betas = np.zeros((n,2))
for i in range(n):
    N = (otsu_thresholds[i] - epsilon) / 255
    C = (otsu_thresholds[i] + epsilon) / 255
    barcode0 = np.loadtxt(os.path.join(barcodes_dir, 'dim0/epoch_%d/dim_0_epoch_%d_data_%d.csv' % (epoch, epoch, i)), delimiter=',')
    beta0 = 0
    for k in range(barcode0.shape[0]):
        pair = barcode0[k, :]
        if pair[0] >= C and pair[1] < N:
            beta0 += 1
    betas[i, 0] = beta0

    barcode1 = np.loadtxt(os.path.join(barcodes_dir, 'dim1/epoch_%d/dim_1_epoch_%d_data_%d.csv' % (epoch, epoch, i)), delimiter=',')
    beta1 = 0
    for k in range(barcode1.shape[0]):
        pair = barcode1[k, :]
        if pair[0] >= C and pair[1] < N:
            beta1 += 1
    betas[i, 1] = beta1
os.makedirs('data/loss_mse/lr_0.0002/alpha', exist_ok=True)
np.savetxt('data/loss_mse/lr_0.0002/alpha/output_betas_epoch_%d.csv' % epoch, betas, fmt='%1.0f', delimiter=',')

