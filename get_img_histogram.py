import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir='images/size85_mini'

all_imgs = []
'''
for img_name in os.listdir(dir):
    img = Image.open(os.path.join(dir, img_name)).convert('L')
    arr = np.asarray(img)
    all_imgs.append(arr)

values = np.stack(all_imgs, axis=0).flatten()'''

img = Image.open(os.path.join(dir, 'data_1470.png')).convert('L')
arr = np.asarray(img)
values = arr.flatten()

print(arr[47:50, :])

f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(values, bins=128)
f.savefig('hist.png')



