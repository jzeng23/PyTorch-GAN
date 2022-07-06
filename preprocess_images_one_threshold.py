import os
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import json

import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

levelsetlayer = LevelSetLayer2D(size=(85, 85), maxdim=1, sublevel=False)

dir = 'images/mini_dataset_resized'
i = 0
betas = []
N = 68
C = 98
barcode_dict = {}
for i in range(len(os.listdir(dir))):

    filename = 'data_' + str(i) + '.png'

    img_original = Image.open(os.path.join(dir, filename))
    img = np.asarray(img_original)
    neighborhood = img.copy()
    neighborhood[neighborhood >= N] = 255
    neighborhood[neighborhood < N] = 0
    
    Image.fromarray(neighborhood).save(os.path.join('images/neighborhood', filename))

    core = img.copy()
    core[core >= C] = 255
    core[core < C] = 0
    Image.fromarray(core).save(os.path.join('images/core', filename))

    diff = img.copy()
    diff[neighborhood == 0] = 0 # set pixels outside of neighborhood and inside core to 0
    diff[core == 255] = 0
    Image.fromarray(diff).save(os.path.join('images/diff', filename))

    # make images with 0, 1/2, 1 values
    new_img = img.copy()
    processed = np.zeros(img.shape)
    processed[new_img >= N] = 0.5
    processed[new_img >= C] = 1
    
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

    barcode_dict[filename] = [barcode0.tolist(), barcode1.tolist()]

    betas.append(row)
    print(str(i), ' ', row)
    processed *= 255
    processed = processed.astype(np.uint8)

    Image.fromarray(processed).save(os.path.join('images/processed', filename))
    
    i += 1

betas = np.stack(betas, axis=0)
np.savetxt('data/betas_mini.csv', betas, fmt='%1.0f', delimiter=',')
print(np.loadtxt('data/betas_mini.csv', delimiter=','))

barcode_json = json.dumps(barcode_dict)
barcode_file = open('implementations/aae/barcodes/ground_truth_mini_dataset.json', 'w')
barcode_file.write(barcode_json)

'''
values = np.stack(arrs, axis=0).flatten()
values = arrs[0].flatten()
f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(values, bins=256)
f.savefig('hist.png')'''