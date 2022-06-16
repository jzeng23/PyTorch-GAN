import os
import numpy as np
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

def threshold(intensity):
    if intensity > 110:
        return 255
    else: 
        return 0

dir = 'data/raw'
i = 0
arrs = []
for img_name in os.listdir(dir):

    img = Image.open(os.path.join(dir, img_name)).convert('L').point(threshold)
    arrs.append(np.asarray(img))
    path = 'images/110/data_' + str(i) + '.png'
    img.save(path)
    img.close()
    i += 1
'''
values = np.stack(arrs, axis=0).flatten()
values = arrs[0].flatten()
f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(values, bins=256)
f.savefig('hist.png')'''