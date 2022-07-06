import numpy as np
from PIL import Image
import os

# augment via rotating original image and mirrored version of image

original_dir = 'data/raw'
save_dir = 'images/full_dataset_different_thresholds/original_resized_augmented'
i = 0

for filename in os.listdir(original_dir):

    im = Image.open(os.path.join(original_dir, filename)).resize((85, 85))
    im.save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1
    im.transpose(Image.ROTATE_90).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1
    im.transpose(Image.ROTATE_180).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1
    im.transpose(Image.ROTATE_270).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1

    im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im.save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1
    im.transpose(Image.ROTATE_90).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1
    im.transpose(Image.ROTATE_180).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1
    im.transpose(Image.ROTATE_270).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
    i += 1

