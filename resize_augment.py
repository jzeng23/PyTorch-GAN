import numpy as np
from PIL import Image
import os
import shutil
import torchvision.transforms


# augment via rotating original image and mirrored version of image

raw_dir = 'data/raw/small'
overall_dir = 'images/full_dataset_different_thresholds'
save_dir = 'images/full_dataset_different_thresholds/original_resized_augmented'
#os.mkdir(save_dir)
i = 0

randomcrop = torchvision.transforms.RandomResizedCrop(size=85, scale=(0.5, 1.0))
list = os.listdir(raw_dir)
print(list)

for filename in list:

    im = Image.open(os.path.join(raw_dir, filename))

    for n in range(150):
        cropped = randomcrop(im)  
        cropped.save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1
        cropped.transpose(Image.ROTATE_90).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1
        cropped.transpose(Image.ROTATE_180).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1
        cropped.transpose(Image.ROTATE_270).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1

        cropped = cropped.transpose(Image.FLIP_LEFT_RIGHT)
        cropped.save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1
        cropped.transpose(Image.ROTATE_90).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1
        cropped.transpose(Image.ROTATE_180).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1
        cropped.transpose(Image.ROTATE_270).save(os.path.join(save_dir, 'data_' + str(i) + '.png'))
        i += 1

N = len(os.listdir(save_dir)) - 1
test_indices = np.random.randint(0, N, size=N//5).tolist()
current_train_index = 0
current_test_index = 0

for j in range(N):
    destination = overall_dir
    if j in test_indices:
        destination += '/test/original/data_' + str(current_test_index) + '.png'
        current_test_index += 1
    else:
        destination += '/train/original/data_' + str(current_train_index) + '.png'
        current_train_index += 1
    os.rename(os.path.join(save_dir, 'data_' + str(j) + '.png'), destination)