import os
import numpy as np
from PIL import Image
import torch

from topologylayer.nn import LevelSetLayer2D

levelsetlayer = LevelSetLayer2D(size=(85, 85), maxdim=1, sublevel=False)

def get_barcode_from_img(path):
    im = np.asarray(Image.open(path)).copy()
    im = torch.Tensor(im) / 255
    barcode = levelsetlayer(im)[0]
    barcode0 = np.asarray(barcode[0])
    np.savetxt('data/original_barcode_dim_0.csv', barcode0, delimiter=',')
    barcode1 = np.asarray(barcode[1])
    np.savetxt('data/original_barcode_dim_1.csv', barcode0, delimiter=',')


path = 'images/different_thresholds/original/data_0.png'
get_barcode_from_img(path)