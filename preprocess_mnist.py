import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

def compute_otsu_criteria(im, th):
    # create the thresholded image
    thresholded_im = np.zeros(im.shape)
    thresholded_im[im >= th] = 1

    # compute weights
    nb_pixels = im.size
    nb_pixels1 = np.count_nonzero(thresholded_im)
    weight1 = nb_pixels1 / nb_pixels
    weight0 = 1 - weight1

    # if one the classes is empty, eg all pixels are below or above the threshold, that threshold will not be considered
    # in the search for the best threshold
    if weight1 == 0 or weight0 == 0:
        return np.inf

    # find all pixels belonging to each class
    val_pixels1 = im[thresholded_im == 1]
    val_pixels0 = im[thresholded_im == 0]

    # compute variance of these classes
    var0 = np.var(val_pixels0) if len(val_pixels0) > 0 else 0
    var1 = np.var(val_pixels1) if len(val_pixels1) > 0 else 0

    return weight0 * var0 + weight1 * var1

levelsetlayer = LevelSetLayer2D(size=(28, 28), maxdim=1, sublevel=False)

def get_betas(img_array):
    img = torch.Tensor(img_array) / 255
    barcode = levelsetlayer(img)[0]
    beta0 = 0
    for pair in barcode[0]:
        length = pair[0] - pair[1]
        if length >= 1.0:
            beta0 += 1
    beta1 = 0
    for pair in barcode[1]:
        length = pair[0] - pair[1]
        if length >= 1.0:
            beta1 += 1 
    ans = [beta0, beta1]
    return ans   

os.makedirs("data/mnist", exist_ok=True)
dataset = datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor()]
        ),
    )
dataset, _ = random_split(dataset, [10, 59990])
dataloader = DataLoader(dataset)
thresholds = np.zeros(10)
epsilon = 15
original_betas = []
ngh_betas = []
core_betas = []
for i, (imgs, _) in enumerate(dataloader):
    im = imgs[0, 0, :, :].numpy()

    original_betas.append(get_betas(im))

    threshold_range = range(np.max(im)+1)
    criterias = [compute_otsu_criteria(im, th) for th in threshold_range]
    best_threshold = threshold_range[np.argmin(criterias)]
    thresholds[i] = best_threshold
    N = (best_threshold - 15) / 255
    C = (best_threshold + 15) / 255

    neighborhood = im.copy()
    neighborhood[neighborhood >= N] = 255
    neighborhood[neighborhood < N] = 0
    neighborhood_scaled = neighborhood / 255
    ngh_betas.append(get_betas(neighborhood_scaled))
    Image.fromarray(neighborhood).save(os.path.join(neighborhood_dir, filename))

    core = im.copy()
    core[core >= C] = 255
    core[core < C] = 0
    core_scaled = core / 255
    core_betas.append(get_betas(core_scaled))
    Image.fromarray(core).save(os.path.join(core_dir, filename))

    