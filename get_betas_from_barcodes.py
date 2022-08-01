import os
import numpy as np
from PIL import Image
from numpy import asarray

import torch

otsu_thresholds = np.loadtxt('data/mini_otsu_thresholds_diff_thresholds.csv', delimiter=',')
goal_betas = np.loadtxt('data/mini_betas_diff_thresholds.csv', delimiter=',')
epsilon = 15
epoch = 6300
barcodes_dir = 'implementations/aae/barcode/loss_mse/lr_0.0002'
images_dir = 'implementations/aae/images/loss_mse/lr_0.0002/epoch_%d' % epoch
core_dir = 'images/different_thresholds/core'
neighborhood_dir = 'images/different_thresholds/neighborhood'
n = 10

pixelwise_loss = torch.nn.MSELoss(reduction='sum')

total_top_loss_0 = 0
total_top_loss_1 = 0
total_core_loss = 0
total_ngh_loss = 0

betas = np.zeros((n,2))
top_losses = np.zeros((n,2))
core_losses = np.zeros(n)
ngh_losses = np.zeros(n)
indices = [0, 1, 2, 6]
for i in range(n):
    N = (otsu_thresholds[i] - epsilon) / 255
    C = (otsu_thresholds[i] + epsilon) / 255
    barcode0 = np.loadtxt(os.path.join(barcodes_dir, 'dim0/epoch_%d/dim_0_epoch_%d_data_%d.csv' % (epoch, epoch, i)), delimiter=',')
    beta0 = 0
    sum0 = 0
    for k in range(barcode0.shape[0]):
        pair = barcode0[k, :]
        if pair[0] >= C and pair[1] < N:
            beta0 += 1
        length = pair[0] - pair[1]
        if length < np.inf:
            sum0 += length
    betas[i, 0] = beta0
    top_loss_0 = (sum0 - goal_betas[i, 0]) ** 2
    top_losses[i, 0] = top_loss_0
    if i in indices:
        total_top_loss_0 += top_loss_0

    barcode1 = np.loadtxt(os.path.join(barcodes_dir, 'dim1/epoch_%d/dim_1_epoch_%d_data_%d.csv' % (epoch, epoch, i)), delimiter=',')
    beta1 = 0
    sum1 = 0
    for k in range(barcode1.shape[0]):
        pair = barcode1[k, :]
        if pair[0] >= C and pair[1] < N:
            beta1 += 1
        sum1 += (pair[0] - pair[1])
    betas[i, 1] = beta1
    top_loss_1 = (sum1 - goal_betas[i, 1]) ** 2
    top_losses[i, 1] = top_loss_1
    if i in indices:
        total_top_loss_1 += top_loss_1

    filename = 'epoch_' + str(epoch) + '_data_' + str(i) + '.png'
    output = np.asarray(Image.open(os.path.join(images_dir, filename)).convert('L')).copy()
    output = output / 255

    filename = 'data_' + str(i) + '.png'
    core = np.asarray(Image.open(os.path.join(core_dir, filename))).copy()
    core = core / 255
    core_size = np.count_nonzero(core)
    output_core = output.copy()
    output_core[core == 0] = 0
    core_loss = np.sum(np.square(core - output_core)) / core_size
    if i in indices:
        total_core_loss += core_loss
    core_losses[i] = core_loss

    neighborhood = np.asarray(Image.open(os.path.join(neighborhood_dir, filename))).copy()
    neighborhood = neighborhood / 255
    neighborhood_complement_size = neighborhood.size - np.count_nonzero(neighborhood)
    output_neighborhood = output.copy()
    output_neighborhood[neighborhood == 1] = 0
    neighborhood_loss = np.sum(np.square(output_neighborhood)) / neighborhood_complement_size
    if i in indices:
        print(i)
        total_ngh_loss += neighborhood_loss
    ngh_losses[i] = neighborhood_loss


os.makedirs('data/loss_mse/lr_0.0002', exist_ok=True)
np.savetxt('data/loss_mse/lr_0.0002/output_betas_epoch_%d.csv' % epoch, betas, fmt='%1.0f', delimiter=',')
np.savetxt('data/loss_mse/lr_0.0002/top_losses_epoch_%d.csv' % epoch, top_losses, delimiter=',')
np.savetxt('data/loss_mse/lr_0.0002/core_losses_epoch_%d.csv' % epoch, core_losses, delimiter=',')
np.savetxt('data/loss_mse/lr_0.0002/ngh_losses_epoch_%d.csv' % epoch, ngh_losses, delimiter=',')
print('mean topological loss: ')
print('dim 0: ', total_top_loss_0 / len(indices))
print('dim 1: ', total_top_loss_1 / len(indices))
print('mean geometry losses: ')
print('core: ', total_core_loss / len(indices))
print('neighborhood: ', total_ngh_loss / len(indices))
