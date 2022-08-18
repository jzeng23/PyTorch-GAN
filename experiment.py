from PIL import Image
import numpy as np

otsu_thresholds = np.loadtxt('data/otsu_thresholds_full_train.csv', delimiter=',')
epsilon = 15
i = 26

im = np.asarray(Image.open('implementations/aae/images/loss_mse/lr_0.0002/alpha/full/epoch_0/epoch_0_data_%d.png' % i)).copy()

N = otsu_thresholds[i] - 15
C = otsu_thresholds[i] + 15

n_img = im.copy()
n_img[n_img >= N] = 255
n_img[n_img < 255] = 0
Image.fromarray(n_img).save('n.png')

c_img = im.copy()
c_img[c_img >= C] = 255
c_img[c_img < 255] = 0
Image.fromarray(c_img).save('c.png')