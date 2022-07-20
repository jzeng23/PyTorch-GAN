# some random experiments to make sure there aren't bugs when preprocessing the images.

import os
import numpy as np
from PIL import Image
from numpy import asarray

otsu_thresholds = np.loadtxt('data/mini_otsu_thresholds_diff_thresholds.csv', delimiter=',')
epsilon = 15

original_dir = 'images/different_thresholds/original'
core_dir = 'images/different_thresholds/core'
neighborhood_dir = 'images/different_thresholds/neighborhood'
processed_dir = 'images/different_thresholds/processed'

n = 10

for i in range(n):
    N = (otsu_thresholds[i] - epsilon) 
    C = (otsu_thresholds[i] + epsilon)
    img_name = 'data_' + str(i) + '.png'
    core = asarray(Image.open(os.path.join(core_dir, img_name)))
    neighborhood = asarray(Image.open(os.path.join(neighborhood_dir, img_name)))
    processed = asarray(Image.open(os.path.join(processed_dir, img_name)))

    processed_core = processed.copy() / 255
    processed_core[processed_core == 1] = 255
    processed_core[processed_core < 1] = 0
    processed_core = processed_core.astype(np.uint8)
    Image.fromarray(processed_core).save(os.path.join('images/processed/core', img_name))
    print(np.array_equal(core, processed_core))

    processed_neighborhood = processed.copy() / 255
    processed_neighborhood[processed_neighborhood >= 127/255] = 255
    processed_neighborhood[processed_neighborhood < 127/255] = 0
    processed_neighborhood = processed_neighborhood.astype(np.uint8)
    Image.fromarray(processed_neighborhood).save(os.path.join('images/processed/neighborhood', img_name))
    print(np.array_equal(neighborhood, processed_neighborhood))


    
    #core = Image.open(os.path.join(core_dir, img_name))
