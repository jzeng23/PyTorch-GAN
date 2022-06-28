import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dir='images/all_images'

avg_img = np.zeros((85, 85))

for img_name in os.listdir(dir):
    img = Image.open(os.path.join(dir, img_name)).convert('L').resize((85, 85))
    arr = np.asarray(img)
    avg_img += arr

avg_img /= len(os.listdir(dir))
values = avg_img.flatten()

f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(values, bins=256)
f.savefig('hist.png')



