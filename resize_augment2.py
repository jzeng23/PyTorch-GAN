import numpy as np
from PIL import Image
import os
import shutil

# augment via rotating original image and mirrored version of image

original_dir = 'data/raw'
overall_dir = 'images/full_dataset_different_thresholds'
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

n = len(os.listdir(save_dir))
test_indices = np.random.randint(0, n, size=n//5).tolist()
current_train_index = 0
current_test_index = 0

for j in range(n):
    destination = overall_dir
    if j in test_indices:
        destination += '/test/original/data_' + str(current_test_index) + '.png'
        current_test_index += 1
    else:
        destination += '/train/original/data_' + str(current_train_index) + '.png'
        current_train_index += 1
    os.rename(os.path.join(save_dir, 'data_' + str(j) + '.png'), destination)