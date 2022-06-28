import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

data = np.loadtxt('latent.csv', delimiter=',')
latent0 = data[:, 0]
latent1 = data[:, 1]

f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(latent0, bins=100)
f.savefig('latent0.png')

f = plt.figure()
plt.title('Image Histogram')
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.hist(latent1, bins=100)
f.savefig('latent1.png')



