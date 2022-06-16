import os
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

levelsetlayer = LevelSetLayer2D(size=(85, 85), maxdim=1, sublevel=False)

dir = 'images/mini_dataset_resized'
i = 0
betas = []
for img_name in os.listdir(dir):

    filename = str(i) + '.png'

    img_original = Image.open(os.path.join(dir, img_name))
    img = np.asarray(img_original)
    neighborhood = img.copy()
    neighborhood[neighborhood >= 80] = 255
    neighborhood[neighborhood < 80] = 0
    
    Image.fromarray(neighborhood).save(os.path.join('images/neighborhood', filename))

    neighborhood_complement = img.copy()
    neighborhood_complement[neighborhood == 0] = 255
    neighborhood_complement[neighborhood == 255] = 0
    Image.fromarray(neighborhood_complement).save(os.path.join('images/nc', filename))

    core = img.copy()
    core[core >= 110] = 255
    core[core < 110] = 0
    Image.fromarray(core).save(os.path.join('images/core', filename))

    # make images with 0, 1/2, 1 values
    new_img = img.copy()
    processed = np.zeros(img.shape)
    processed[new_img >= 80] = 0.5
    processed[new_img >= 110] = 1

    row = np.zeros(2)
    barcode = levelsetlayer(torch.Tensor(processed))[0]
    barcode0 = barcode[0]
    goal_b0 = 0
    for bar in barcode0:
        if bar[0] == 1 and bar[1] == 0.5:
            goal_b0 += 1
    row[0] = goal_b0

    barcode1 = barcode[1]
    goal_b1 = 0
    for bar in barcode1:
        if bar[0] == 1 and bar[1] == 0.5:
            goal_b1 += 1
    row[1] = goal_b1

    betas.append(row)

    processed *= 255
    processed = processed.astype(np.uint8)

    Image.fromarray(processed).save(os.path.join('images/processed', filename))
    
    i += 1

betas = np.stack(betas, axis=0)
np.savetxt('betas_mini.csv', betas, delimiter=',')
'''
values = np.stack(arrs, axis=0).flatten()
values = arrs[0].flatten()
f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(values, bins=256)
f.savefig('hist.png')'''