import os
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import json

import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

levelsetlayer = LevelSetLayer2D(size=(85, 85), maxdim=1, sublevel=False)
epsilon = 15

def get_betas_core_ngh(dir):
    N = len(os.listdir(dir))
    betas = np.zeros((N, 2))
    for i in range(N):
        img_name = 'data_' + str(i) + '.png'
        img = np.asarray(Image.open(os.path.join(dir, img_name))).copy()
        
        img = torch.Tensor(img) / 255
        barcode = levelsetlayer(img)[0]

        barcode0 = barcode[0]
        beta0 = 0
        for pair in barcode0:
            if pair[0] - pair[1] == 1:
                beta0 += 1
        betas[i, 0] = beta0

        barcode1 = barcode[1]
        beta1 = 0
        for pair in barcode1:
            if pair[0] - pair[1] == 1:
                beta1 += 1
        betas[i, 1] = beta1

    return betas

def get_betas(dir, thresholds):
    N = len(os.listdir(dir))
    betas = np.zeros((N, 2))
    for i in range(N):
        img_name = 'data_' + str(i) + '.png'
        img = np.asarray(Image.open(os.path.join(dir, img_name))).copy()
        T = thresholds[i]
        N = (T - epsilon) / 255
        C = (T + epsilon) / 255

        img = torch.Tensor(img) / 255
        barcode = levelsetlayer(img)[0]
        barcode0 = barcode[0]
        beta0 = 0
        for pair in barcode0:
            if pair[0] >= C and pair[1] < N:
                beta0 += 1
        betas[i, 0] = beta0

        barcode1 = barcode[1]
        beta1 = 0
        for pair in barcode1:
            if pair[0] >= C and pair[1] < N:
                beta1 += 1
        betas[i, 1] = beta1
    return betas

otsu_thresholds = np.loadtxt('data/mini_otsu_thresholds_diff_thresholds.csv', delimiter=',')
dir = 'images/different_thresholds/original'
betas = get_betas(dir, otsu_thresholds)
np.savetxt('data/mini_betas_raw.csv', betas, fmt='%1.0f', delimiter=',')

