'''
current settings:
weights: 3*ngh + 4*core + 0.001*top0 + 0.001*top1 + 0.001 adversarial
loss: absolute value
'''

import argparse
import os
from re import M
from statistics import variance
import numpy as np
import math
import itertools
from PIL import Image
from scipy.fftpack import diff
import time
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torch.autograd import Variable
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import torch.nn.functional as F
import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8192, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=40, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=85, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--decoder_input_channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--save_dir", type=str, default='latest_model/loss_mse/ExponentialLR/0.0002/gamma_0.9999/alpha_1.0003/train_50000', help="directory where you save models")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
settings = 'loss_mse/ExponentialLR/0.0002/gamma_0.9999/alpha_1.0003/train_50000'
os.makedirs(opt.save_dir, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

levelsetlayer = LevelSetLayer2D(size=(opt.img_size, opt.img_size), maxdim=1, sublevel=False)
beta0layer = SumBarcodeLengths(dim=0)
beta1layer = SumBarcodeLengths(dim=1)
alpha = 1

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

class ImageSet(Dataset):
  def __init__(self, img_dir, n_dir, c_dir, betas_path, thresholds_path, transform=None):
    self.img_nums = []
    self.img_dir = img_dir
    self.n_dir = n_dir
    self.c_dir = c_dir
    self.transform = transform
    self.imgs = []
    self.betas = np.loadtxt(betas_path, delimiter=',')
    self.neighborhoods = []
    self.cores = []
    self.nc_sizes = []
    self.core_sizes = []
    self.thresholds = np.loadtxt(thresholds_path, delimiter=',')
    for index in range(len(os.listdir(img_dir))):
        filename = 'data_' + str(index) + '.png'
        self.img_nums.append(index)
        img = Image.open(os.path.join(img_dir, filename))
        img_transformed = self.transform(img)
        self.imgs.append(img_transformed)
        img.close()

        neighborhood = Image.open(os.path.join(n_dir, filename))
        neighborhood_transformed = self.transform(neighborhood)
        nc_size = opt.img_size * opt.img_size - torch.count_nonzero(neighborhood_transformed)
        self.nc_sizes.append(nc_size)
        self.neighborhoods.append(neighborhood_transformed)
        neighborhood.close()

        core = Image.open(os.path.join(c_dir, filename))
        core_transformed = self.transform(core)
        core_size = torch.count_nonzero(core_transformed)
        self.core_sizes.append(core_size)
        self.cores.append(core_transformed)
        core.close()
        
  def __getitem__(self, index):
    img_num = self.img_nums[index]
    img = self.imgs[index]
    neighborhood = self.neighborhoods[index]
    core = self.cores[index]
    beta0 = self.betas[index, 0]
    beta1 = self.betas[index, 1]
    nc_size = self.nc_sizes[index]
    core_size = self.core_sizes[index]
    threshold = self.thresholds[index]
    return img_num, img, neighborhood, core, beta0, beta1, nc_size, core_size, threshold

  def __len__(self):
    return len(self.imgs)

class Encoder(nn.Module):
    def __init__(self, input_size, step_channels=64, nonlinearity=nn.LeakyReLU(0.2)):
        super(Encoder, self).__init__()
        size = input_size
        channels = step_channels
        encoder = [
            nn.Sequential(
                nn.Conv2d(1, step_channels, 5, 4, 0), nonlinearity
            )
        ]
        size = (size - 5) // 4 + 1
        while size > 1:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels * 4, 5, 4, 0),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size = (size - 5) // 4 + 1
        self.encoder = nn.Sequential(*encoder)
        intermediate = (channels + opt.latent_dim) // 2
        self.encoder_fc = nn.Sequential(
            nn.Linear(channels, intermediate),
            nn.Linear(intermediate, opt.latent_dim)
        )

    def forward(self, img):
        x = self.encoder(img)
        channels = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, channels)
        return self.encoder_fc(x)


class Decoder(nn.Module):
    def __init__(self, input_size, step_channels=1024, nonlinearity=nn.LeakyReLU(0.2)):
        super(Decoder, self).__init__()

        intermediate = (step_channels + opt.latent_dim) // 2
        self.decoder_fc = nn.Sequential(
            nn.Linear(opt.latent_dim, intermediate),
            nn.Linear(intermediate, step_channels)
        )

        decoder = []
        size = 1
        channels = step_channels

        while size < (input_size - 5) // 4 + 1:
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels // 4, 5, 4, 0),
                    nn.BatchNorm2d(channels // 4),
                    nonlinearity,
                )
            )
            channels = channels // 4
            size = 4*(size - 1) + 5

        decoder.append(nn.ConvTranspose2d(channels, 1, 5, 4, 0))
        self.decoder = nn.Sequential(*decoder)

        self.sigmoid = nn.Sigmoid()

    def forward(self, z, alpha):
        z = self.decoder_fc(z)
        z = z.view(-1, z.size(1), 1, 1)
        z = self.decoder(z)
        return self.sigmoid(alpha*z)


class Discriminator(nn.Module):
    def __init__(self, input_dims, nonlinearity=nn.LeakyReLU(0.2)):
        super(Discriminator, self).__init__()

        model = [nn.Sequential(nn.Linear(input_dims, input_dims // 2), nonlinearity)]
        size = input_dims // 2
        while size > 128:
            model.append(
                nn.Sequential(
                    nn.Linear(size, size // 2), nn.BatchNorm1d(size // 2), nonlinearity
                )
            ) 
            size = size // 2
        model.append(nn.Linear(size, 1))
        self.model = nn.Sequential(*model)


    def forward(self, z):
        validity = self.model(z)
        return validity


# Use mean squared error
adversarial_loss = torch.nn.MSELoss(reduction='mean')
pixelwise_loss = torch.nn.MSELoss(reduction='sum')

# Initialize generator and discriminator
encoder = Encoder(input_size=85)
decoder = Decoder(input_size=85)
discriminator = Discriminator(input_dims=opt.latent_dim)

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Configure data loaders
transform=transforms.Compose([transforms.ToTensor(),])
dataset =ImageSet(
    img_dir='/scratch/train_topological_aae/n_63670/original',
    n_dir='/scratch/train_topological_aae/n_63670/neighborhood',
    c_dir='/scratch/train_topological_aae/n_63670/core',
    betas_path='/scratch/train_topological_aae/n_63670/betas_train_50000_train.csv',
    thresholds_path='/scratch/train_topological_aae/n_63670/otsu_thresholds_train_50000.csv',
    transform=transform)

valid_size = len(dataset) // 5
train_size = len(dataset) - valid_size
train, valid = random_split(dataset, [train_size, valid_size])
trainloader = torch.utils.data.DataLoader(train, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=20)
validloader = torch.utils.data.DataLoader(valid, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=20)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.9999)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.9999)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_image(current_epoch, imgs, nums):
    """Saves a grid of training data before and after it's put through the autoencoder"""
    # Sample noise
    img_save_path = 'images/' + settings + '/epoch_%d' % current_epoch

    os.makedirs(img_save_path, exist_ok=True)

    for i in range(nums.size(0)):
        index = nums[i]
        img = imgs[i, 0, :, :]
        save_image(img, os.path.join(img_save_path, 'epoch_%d_data_%d.png' % (current_epoch, index)))

def get_plots(current_epoch, losses, goals, actuals, sums):
    save_folder = 'plots/' + settings + '/epoch_%d' % current_epoch
    os.makedirs(save_folder, exist_ok=True)

    f = plt.figure()
    plt.title('MSE of Sum vs Goal - Dim 0')
    plt.ylabel('Frequency')
    plt.xlabel('value')
    plt.hist(losses[:, 0], bins=20)
    f.savefig(os.path.join(save_folder, 'losses_0.png'))

    f = plt.figure()
    plt.title('MSE of Sum vs Goal - Dim 1')
    plt.ylabel('Frequency')
    plt.xlabel('value')
    plt.hist(losses[:, 1], bins=20)
    f.savefig(os.path.join(save_folder, 'losses_1.png'))
    
    f = plt.figure()
    plt.title('Actual Number of Artifacts vs Goal - Dim 0')
    plt.scatter(goals[:, 0], actuals[:, 0])
    plt.xlabel('Goal')
    plt.ylabel('Actual')
    f.savefig(os.path.join(save_folder, 'actual_scatter_0.png'))

    f = plt.figure()
    plt.title('Actual Number of Artifacts vs Goal - Dim 1')
    plt.scatter(goals[:, 1], actuals[:, 1])
    plt.xlabel('Goal')
    plt.ylabel('Actual')
    f.savefig(os.path.join(save_folder, 'actual_scatter_1.png'))

    f = plt.figure()
    plt.title('Sums vs Goal - Dim 0')
    plt.scatter(goals[:, 0], sums[:, 0])
    plt.xlabel('Goal')
    plt.ylabel('Sum')
    f.savefig(os.path.join(save_folder, 'sum_scatter_0.png'))

    f = plt.figure()
    plt.title('Sums vs Goal - Dim 1')
    plt.scatter(goals[:, 1], sums[:, 1])
    plt.xlabel('Goal')
    plt.ylabel('Sum')
    f.savefig(os.path.join(save_folder, 'sum_scatter_1.png'))
    
def save_models(current_epoch, losses):
    torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            #'scheduler_G_state_dict': scheduler_G.state_dict(),
            #'learning_rate': scheduler_G.get_last_lr()[0],
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'epoch': current_epoch,
            'loss_generator': losses['loss_generator'],
            'loss_discriminator': losses['loss_discriminator'],
            'loss_core': losses['loss_core'],
            'loss_neighborhood': losses['loss_neighborhood'],
            'loss_top_0': losses['loss_top_0'],
            'loss_top_1': losses['loss_top_1'],
            }, os.path.join(opt.save_dir, 'epoch_%d.tar' % current_epoch))

def sample_barcode(imgs, nums, epoch):
    dim0_path = 'barcode/' + settings + '/dim0/epoch_%d' % epoch
    dim1_path = 'barcode/' + settings + '/dim1/epoch_%d' % epoch
    os.makedirs(dim0_path, exist_ok=True)
    os.makedirs(dim1_path, exist_ok=True)
    for i in range(nums.size(0)):
        index = nums[i]
        im = imgs[i, 0, :, :]
        barcode = levelsetlayer(im)[0]
        barcode0 = np.asarray(barcode[0].cpu())
        np.savetxt(os.path.join(dim0_path, 'dim_0_epoch_%d_data_%d.csv' % (epoch, index)), barcode0, delimiter=',')
        barcode1 = np.asarray(barcode[1].cpu())
        np.savetxt(os.path.join(dim1_path, 'dim_1_epoch_%d_data_%d.csv' % (epoch, index)), barcode1, delimiter=',')

def sample_image_random_noise(current_epoch, n):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n, opt.latent_dim))))
    gen_imgs = decoder(z, 1)
    os.makedirs('images/' + settings + '/epoch_%d/random_noise' % current_epoch, exist_ok=True)
    for i in range(n):
        save_path = 'images/' + settings + '/epoch_%d/random_noise/random_%d_epoch_%d.png' % (current_epoch, i, current_epoch)
        save_image(gen_imgs[i, 0, :, :], save_path)

# variables for tracking epoch with best validation performance
best_epoch = 0
best_val_loss = np.inf

# initialize SummaryWriter
writer = SummaryWriter()

# ----------
#  Training
# ----------
encoder.train()
decoder.train()
discriminator.train()
for epoch in range(opt.n_epochs):
    epoch_g_loss = 0
    epoch_top_loss_0 = 0
    epoch_top_loss_1 = 0
    epoch_d_loss = 0
    epoch_c_loss = 0
    epoch_n_loss = 0
    epoch_diff_loss = 0
    learning_rate = 0
    epoch_barcodes = {}
    for i, (img_nums, imgs, neighborhoods, cores, train_beta0_goal, train_beta1_goal, train_nc_size, train_core_size, _) in enumerate(trainloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        n_masks = Variable(neighborhoods.type(Tensor))
        c_masks = Variable(cores.type(Tensor))
        beta0_goal = Variable(train_beta0_goal.type(Tensor))
        beta1_goal = Variable(train_beta1_goal.type(Tensor))
        train_nc_size = Variable(train_nc_size.type(Tensor))
        train_core_size = Variable(train_core_size.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs, alpha)
        # decoded_imgs = tensor of shape [64, 1, 32, 32]

        # calculate neighborhood loss (how much is not in the neighborhood)
        decoded_n_complement = decoded_imgs.clone()
        decoded_n_complement[n_masks == 1] = 0 # mean square of pixels in decoded_n_complement

        # calculate core loss (how much of core is not contained)
        decoded_core = decoded_imgs.clone()
        decoded_core[c_masks == 0] = 0 # mean (1-a)^2 for each pixel value a in decoded_core

        core_loss = 0
        ngh_loss = 0
        
        core_squared_diffs = torch.square(torch.sub(c_masks, decoded_core))
        core_loss_per_image = torch.sum(torch.sum(core_squared_diffs, dim=2), dim=2)
        core_loss_per_image = core_loss_per_image[:, 0]
        avg_core_loss_per_image = core_loss_per_image / train_core_size
        core_loss = 4*torch.sum(avg_core_loss_per_image) / decoded_imgs.size(0)

        nc_squared = torch.square(decoded_n_complement)
        ngh_loss_per_image = torch.sum(torch.sum(nc_squared, dim=2), dim=2)
        ngh_loss_per_image = ngh_loss_per_image[:, 0]
        avg_ngh_loss_per_image = ngh_loss_per_image / train_nc_size
        ngh_loss = 3*torch.sum(avg_ngh_loss_per_image) / decoded_imgs.size(0)

        epoch_n_loss += ngh_loss.item()
        epoch_c_loss += core_loss.item()

        # calculate topological loss
        get_barcode = 0
        get_sums = 0
        top_loss_0 = 0
        top_loss_1 = 0
        for n in range(decoded_imgs.size(0)):
            decoded = decoded_imgs[n, 0, :, :]
            target0 = beta0_goal[n]
            target1 = beta1_goal[n]
            barcode = levelsetlayer(decoded)
            beta0 = beta0layer(barcode)
            beta1 = beta1layer(barcode)
            top_loss_0 += (target0 - beta0)**2
            top_loss_1 += (target1 - beta1)**2
        top_loss_0 = 0.00001 * top_loss_0 / decoded_imgs.size(0)
        top_loss_1 = 0.00001 * top_loss_1 / decoded_imgs.size(0)
        epoch_top_loss_0 += top_loss_0
        epoch_top_loss_1 += top_loss_1

        # Loss measures generator's ability to fool the discriminator
        g_loss = ngh_loss + core_loss + top_loss_0 + top_loss_1 + 0.001 * adversarial_loss(discriminator(encoded_imgs), valid)
        epoch_g_loss += g_loss.item()
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        epoch_d_loss += d_loss.item()
        d_loss.backward()
        optimizer_D.step()
        print(
            "[Epoch %d/%d] Training [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(trainloader), d_loss.item(), g_loss.item())
        )
        if epoch % opt.sample_interval == 0 and i == 0:
            sample_image(current_epoch=epoch, imgs=decoded_imgs[0:10, :, :, :], nums=img_nums[0:10])
            #sample_barcode(decoded_imgs[0:10, :, :, :], img_nums[0:10] epoch)

    learning_rate = scheduler_G.get_last_lr()[0]
    scheduler_G.step()
    scheduler_D.step()
    alpha *= 1.0003
    
    # -----------
    # Validation
    # -----------
    
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    epoch_valid_g_loss = 0
    epoch_valid_c_loss = 0
    epoch_valid_n_loss = 0
    epoch_valid_top_loss_0 = 0
    epoch_valid_top_loss_1 = 0
    goals = []
    actuals = []
    losses = []
    sums = []
    for j, (_, valid_imgs, valid_neighborhoods, valid_cores, valid_beta0_goal, valid_beta1_goal, valid_nc_size, valid_core_size, valid_thresholds) in enumerate(validloader):

        # Adversarial ground truths
        valid = Variable(Tensor(valid_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(valid_imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(valid_imgs.type(Tensor))
        n_masks = Variable(valid_neighborhoods.type(Tensor))
        c_masks = Variable(valid_cores.type(Tensor))
        beta0_goal = Variable(valid_beta0_goal.type(Tensor))
        beta1_goal = Variable(valid_beta1_goal.type(Tensor))
        valid_nc_size = Variable(valid_nc_size.type(Tensor))
        valid_core_size = Variable(valid_core_size.type(Tensor))

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs, 1)
        # decoded_imgs = tensor of shape [64, 1, 32, 32]

        # calculate neighborhood loss (how much is not in the neighborhood)
        decoded_n_complement = decoded_imgs.clone()
        decoded_n_complement[n_masks == 1] = 0 # mean square of pixels in decoded_n_complement

        # calculate core loss (how much of core is not contained)
        decoded_core = decoded_imgs.clone()
        decoded_core[c_masks == 0] = 0 # mean (1-a)^2 for each pixel value a in decoded_core

        core_loss = 0
        ngh_loss = 0

        core_squared_diffs = torch.square(torch.sub(c_masks, decoded_core))
        core_loss_per_image = torch.sum(torch.sum(core_squared_diffs, dim=2), dim=2)
        core_loss_per_image = core_loss_per_image[:, 0]
        avg_core_loss_per_image = core_loss_per_image / valid_core_size
        core_loss = 4*torch.sum(avg_core_loss_per_image) / decoded_imgs.size(0)

        nc_squared = torch.square(decoded_n_complement)
        ngh_loss_per_image = torch.sum(torch.sum(nc_squared, dim=2), dim=2)
        ngh_loss_per_image = ngh_loss_per_image[:, 0]
        avg_ngh_loss_per_image = ngh_loss_per_image / valid_nc_size
        ngh_loss = 3*torch.sum(avg_ngh_loss_per_image) / decoded_imgs.size(0)

        epoch_valid_c_loss += core_loss.item()
        epoch_valid_n_loss += ngh_loss.item()

        # calculate topological loss
        top_loss = 0
        batch_losses = np.zeros((decoded_imgs.size(0), 2))
        batch_sums = np.zeros((decoded_imgs.size(0), 2))
        batch_actuals = np.zeros((decoded_imgs.size(0), 2))
        for n in range(decoded_imgs.size(0)):
            decoded = decoded_imgs[n, 0, :, :]
            target0 = beta0_goal[n]
            target1 = beta1_goal[n]
            barcode = levelsetlayer(decoded)
            beta0 = beta0layer(barcode)
            beta1 = beta1layer(barcode)
            top_loss_0 += (target0 - beta0)**2
            top_loss_1 += (target1 - beta1)**2
            if epoch % opt.sample_interval == 0:
                batch_losses[n, 0] = (target0 - beta0)**2
                batch_losses[n, 1] = (target1 - beta1)**2
                batch_sums[n, 0] = beta0
                batch_sums[n, 1] = beta1

                N = (valid_thresholds[n] - 15) / 255
                C = (valid_thresholds[n] + 15) / 255

                barcode0 = barcode[0][0]
                birth0 = barcode0[:, 0]
                death0 = barcode0[:, 1]
                death0 = death0[birth0 >= C]
                death0 = death0[death0 < N]
                batch_actuals[n, 0] = death0.size(0)

                barcode1 = barcode[0][1]
                birth1 = barcode1[:, 0]
                death1 = barcode1[:, 1]
                death1 = death1[birth1 >= C]
                death1 = death1[death1 < N]
                batch_actuals[n, 1] = death1.size(0)
                
        top_loss_0 = 0.00001 * top_loss_0 / decoded_imgs.size(0)
        top_loss_1 = 0.00001 * top_loss_1 / decoded_imgs.size(0)
        epoch_valid_top_loss_0 += top_loss_0
        epoch_valid_top_loss_1 += top_loss_1

        # Loss measures generator's ability to fool the discriminator
        valid_g_loss = ngh_loss + core_loss + top_loss_0 + top_loss_1 + 0.001 * adversarial_loss(discriminator(encoded_imgs), valid)
        epoch_valid_g_loss += valid_g_loss.item()

        if epoch % opt.sample_interval == 0:
            batch_goals = torch.stack((valid_beta0_goal.clone().detach(), valid_beta1_goal.clone().detach()), 1)
            if j == 0:
                goals = batch_goals
                losses = batch_losses
                sums = batch_sums
                actuals = batch_actuals
            else:
                goals = torch.cat((goals, batch_goals), 0)
                losses = np.concatenate((losses, batch_losses), axis=0)
                sums = np.concatenate((sums, batch_sums), axis=0)
                actuals = np.concatenate((actuals, batch_actuals), axis=0)

        print(
            "[Epoch %d/%d] Validation [Batch %d/%d] [G loss: %f]"
            % (epoch, opt.n_epochs, j, len(validloader), valid_g_loss.item())
        )
    
    epoch_valid_g_loss /= len(validloader)
    if epoch_valid_g_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = epoch_valid_g_loss

    if epoch % opt.sample_interval == 0:
        get_plots(epoch, losses, goals, actuals, sums)
        sample_image_random_noise(epoch, 10)
    
         
    writer.add_scalars('ExponentialLR(0.0002, 0.9999), MSE 1e-5, new goal betas, 50936 train images (not including validation)', {
        'Train Generator': epoch_g_loss / len(trainloader),
        'Train Discriminator': epoch_d_loss / len(trainloader),
        'Train Core' : epoch_c_loss / len(trainloader),
        'Train Ngh': epoch_n_loss / len(trainloader),
        'Train Top 0': epoch_top_loss_0 / len(trainloader),
        'Train Top 1': epoch_top_loss_1 / len(trainloader)
    }, epoch)

    if epoch % opt.sample_interval == 0:
        losses = {
            'loss_generator': epoch_g_loss / len(trainloader),
            'loss_discriminator': epoch_d_loss / len(trainloader),
            'loss_core' : epoch_c_loss / len(trainloader),
            'loss_neighborhood': epoch_n_loss / len(trainloader),
            'loss_top_0': epoch_top_loss_0 / len(trainloader),
            'loss_top_1': epoch_top_loss_1 / len(trainloader)
        }
        save_models(epoch, losses)
    
    writer.add_scalars('Validation Loss', {
        'Valid Generator': epoch_valid_g_loss / len(validloader),
        'Valid Core' : epoch_valid_c_loss / len(validloader),
        'Valid Ngh': epoch_valid_n_loss / len(validloader),
        'Valid Top 0': epoch_valid_top_loss_0 / len(validloader),
        'Valid Top 1': epoch_valid_top_loss_1 / len(validloader)
    }, epoch)

    writer.add_scalars('Learning Rate', {'LR': learning_rate}, epoch)
    
