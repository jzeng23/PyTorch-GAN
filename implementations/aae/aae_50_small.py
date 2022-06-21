import argparse
import os
from re import M
import numpy as np
import math
import itertools
from PIL import Image

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

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--a", type=float, default=50000, help="weight of topology loss")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=500, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=85, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--decoder_input_channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1, help="interval between image sampling")
parser.add_argument("--save_dir", type=str, default='best_model/retina', help="directory where you save models")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

class ImageSet(Dataset):
  def __init__(self, img_dir, n_dir, c_dir, diff_dir, csv_path, transform=None):
    self.img_dir = img_dir
    self.n_dir = n_dir
    self.c_dir = c_dir
    self.transform = transform
    self.betas = Tensor(np.loadtxt(csv_path, delimiter=','))
    self.imgs = []
    self.neighborhoods = []
    self.cores = []
    self.diffs = []
    for filename in os.listdir(img_dir):
        img = Image.open(os.path.join(img_dir, filename))
        img_transformed = self.transform(img)
        self.imgs.append(img_transformed)
        img.close()

    for filename in os.listdir(n_dir):
        neighborhood = Image.open(os.path.join(n_dir, filename))
        neighborhood_transformed = self.transform(neighborhood)
        self.neighborhoods.append(neighborhood_transformed)
        neighborhood.close()

    for filename in os.listdir(c_dir):
        core = Image.open(os.path.join(c_dir, filename))
        core_transformed = self.transform(core)
        self.cores.append(core_transformed)
        core.close()

    for filename in os.listdir(diff_dir):
        diff = Image.open(os.path.join(diff_dir, filename))
        diff_transformed = self.transform(diff)
        self.diffs.append(diff_transformed)
        diff.close()

  def __getitem__(self, index):
    img = self.imgs[index]
    neighborhood = self.neighborhoods[index]
    core = self.cores[index]
    diff = self.diffs[index]
    beta0 = self.betas[index, 0]
    beta1 = self.betas[index, 1]
    return img, neighborhood, core, diff, beta0, beta1

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

        self.encoder_fc = nn.Sequential(
            nn.Linear(channels, opt.latent_dim)
        )

    def forward(self, img):
        x = self.encoder(img)
        channels = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, channels)
        return self.encoder_fc(x)


class Decoder(nn.Module):
    def __init__(self, input_size, step_channels=64, nonlinearity=nn.LeakyReLU(0.2)):
        super(Decoder, self).__init__()

        self.decoder_fc = nn.Linear(opt.latent_dim, step_channels)

        decoder = []
        size = 1
        channels = step_channels

        while size < (input_size - 5) // 4 + 1:
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels, channels * 4, 5, 4, 0),
                    nn.BatchNorm2d(channels * 4),
                    nonlinearity,
                )
            )
            channels *= 4
            size = 4*(size - 1) + 5

        decoder.append(nn.ConvTranspose2d(channels, 1, 5, 4, 0))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, z):
        z = self.decoder_fc(z)
        z = z.view(-1, z.size(1), 1, 1)
        return self.decoder(z)


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
pixelwise_loss = torch.nn.MSELoss(reduction='mean')

# Initialize generator and discriminator
encoder = Encoder(input_size=85)
decoder = Decoder(input_size=85, step_channels=opt.decoder_input_channels)
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
    img_dir='../../images/mini_dataset_resized',
    n_dir='../../images/neighborhood',
    c_dir='../../images/core',
    diff_dir = '../../images/diff',
    csv_path='../../data/betas_mini.csv',
    transform=transform)
valid_size = len(dataset) // 5
train_size = len(dataset) - valid_size
train, valid = random_split(dataset, [train_size, valid_size])
trainloader = torch.utils.data.DataLoader(train, batch_size=opt.batch_size, shuffle=True)
validloader = torch.utils.data.DataLoader(valid, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

levelsetlayer = LevelSetLayer2D(size=(opt.img_size, opt.img_size), maxdim=1, sublevel=False)
beta0layer = SumBarcodeLengths(dim=0)

def sample_image(n_row, batches_done, current_epoch):
    """Saves a grid of generated digits"""
    # Sample noise
    z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "images/images_retina/epoch_%d.png" % current_epoch, nrow=n_row, normalize=True)


# initialize SummaryWriter
writer = SummaryWriter()

# variables for tracking epoch with best validation performance
best_epoch = 0
best_val_loss = np.inf

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    epoch_g_loss = 0
    epoch_top_loss = 0
    epoch_d_loss = 0
    encoder.train()
    decoder.train()
    discriminator.train()
    for i, (imgs, neighborhoods, cores, diffs, beta0_goal, beta1_goal) in enumerate(trainloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        n_masks = Variable(neighborhoods.type(Tensor))
        c_masks = Variable(cores.type(Tensor))
        d_masks = Variable(diffs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        # decoded_imgs = tensor of shape [64, 1, 32, 32]

        # calculate diff loss (pixels in neighborhood, outside core)
        decoded_diff = decoded_imgs.clone().detach()
        decoded_diff[d_masks == 0] = 0 # set pixels not in diff to 0
        print(decoded_diff.is_cuda, d_masks.is_cuda)
        diff_loss = pixelwise_loss(decoded_diff, d_masks)

        # calculate neighborhood loss (how much is not in the neighborhood)
        decoded_n_complement = decoded_imgs.clone().detach()
        decoded_n_complement[n_masks == 1] = 0
        ngh_loss = pixelwise_loss(decoded_n_complement, torch.zeros_like(decoded_n_complement)) # mean square of pixels in decoded_n_complement

        # calculate core loss (how much of core is not contained)
        decoded_core = decoded_imgs.clone().detach()
        decoded_core[c_masks == 0] = 0
        core_loss = pixelwise_loss(decoded_core, c_masks) # mean (1-a)^2 for each pixel value a in decoded_core

        # calculate betas of decoded imgs
        top_loss = 0
        T = 95/255
        epsilon = 15/255
        n_threshold = T - epsilon
        c_threshold = T + epsilon
        
        for n in range(decoded_imgs.size(0)):
            decoded = decoded_imgs[n, 0, :, :]
            decoded[decoded < n_threshold] = 0
            mask = decoded >= c_threshold
            decoded[decoded >= n_threshold] = 0.5
            decoded[mask] = 1
            barcode = levelsetlayer(decoded)[0]

            beta0_target = beta0_goal[n]
            barcode0 = barcode[0]
            barcode0_births = barcode0[:, 0]
            barcode0_deaths = barcode0[:, 1]
            barcode0_deaths = barcode0_deaths[barcode0_births == 1]
            barcode0_deaths = barcode0_deaths[barcode0_deaths == 0.5]
            beta0 = barcode0_deaths.size(0)
            top_loss += (beta0_target - beta0) ** 2

            beta1_target = beta1_goal[n]
            barcode1 = barcode[1]
            barcode1_births = barcode1[:, 0]
            barcode1_deaths = barcode1[:, 1]
            barcode1_deaths = barcode1_deaths[barcode1_births == 1]
            barcode1_deaths = barcode1_deaths[barcode1_deaths == 0.5]
            beta1 = barcode1_deaths.size(0)
            top_loss += (beta1_target - beta1) ** 2

        top_loss = top_loss / decoded_imgs.size(0)
        epoch_top_loss += top_loss

        # Loss measures generator's ability to fool the discriminator
        g_loss = 2 * diff_loss + ngh_loss + core_loss + 0.00001 * top_loss + 0.001 * adversarial_loss(discriminator(encoded_imgs), valid)
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

        batches_done = epoch * len(trainloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done, current_epoch=epoch)

    # -----------
    # Validation
    # -----------
    encoder.eval()
    decoder.eval()
    discriminator.eval()
    epoch_valid_g_loss = 0
    epoch_valid_top_loss = 0
    epoch_valid_d_loss = 0
    for j, (valid_imgs, valid_neighborhoods, valid_cores, valid_diffs, valid_beta0_goal, valid_beta1_goal) in enumerate(validloader):

        valid = Variable(Tensor(valid_imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(valid_imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(valid_imgs.type(Tensor))
        n_masks = Variable(valid_neighborhoods.type(Tensor))
        c_masks = Variable(valid_cores.type(Tensor))
        d_masks = Variable(valid_diffs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)
        # decoded_imgs = tensor of shape [64, 1, 32, 32]

        # calculate diff loss (pixels in neighborhood, outside core)
        decoded_diff = decoded_imgs.clone().detach()
        decoded_diff[d_masks == 0] = 0 # set pixels not in diff to 0
        diff_loss = pixelwise_loss(decoded_diff, d_masks)

        # calculate neighborhood loss (how much is not in the neighborhood)
        decoded_n_complement = decoded_imgs.clone().detach()
        decoded_n_complement[n_masks == 1] = 0
        ngh_loss = pixelwise_loss(decoded_n_complement, torch.zeros_like(decoded_n_complement)) # mean square of pixels in decoded_n_complement

        # calculate core loss (how much of core is not contained)
        decoded_core = decoded_imgs.clone().detach()
        decoded_core[c_masks == 0] = 0
        core_loss = pixelwise_loss(decoded_core, c_masks) # mean (1-a)^2 for each pixel value a in decoded_core

        # calculate betas of decoded imgs
        top_loss = 0
        T = 95/255
        epsilon = 15/255
        n_threshold = T - epsilon
        c_threshold = T + epsilon
        
        for m in range(decoded_imgs.size(0)):
            decoded = decoded_imgs[m, 0, :, :]
            decoded[decoded < n_threshold] = 0
            mask = decoded >= c_threshold
            decoded[decoded >= n_threshold] = 0.5
            decoded[mask] = 1
            barcode = levelsetlayer(decoded)[0]

            beta0_target = beta0_goal[m]
            barcode0 = barcode[0]
            barcode0_births = barcode0[:, 0]
            barcode0_deaths = barcode0[:, 1]
            barcode0_deaths = barcode0_deaths[barcode0_births == 1]
            barcode0_deaths = barcode0_deaths[barcode0_deaths == 0.5]
            beta0 = barcode0_deaths.size(0)
            top_loss += (beta0_target - beta0) ** 2

            beta1_target = beta1_goal[m]
            barcode1 = barcode[1]
            barcode1_births = barcode1[:, 0]
            barcode1_deaths = barcode1[:, 1]
            barcode1_deaths = barcode1_deaths[barcode1_births == 1]
            barcode1_deaths = barcode1_deaths[barcode1_deaths == 0.5]
            beta1 = barcode1_deaths.size(0)
            top_loss += (beta1_target - beta1) ** 2

        top_loss = top_loss / decoded_imgs.size(0)
        epoch_top_loss += top_loss

        # Loss measures generator's ability to fool the discriminator
        valid_g_loss = 2 * diff_loss + ngh_loss + core_loss + 0.00001 * top_loss + 0.001 * adversarial_loss(discriminator(encoded_imgs), valid)
        epoch_valid_g_loss += valid_g_loss.item()

    epoch_valid_g_loss /= len(validloader)
    if epoch_valid_g_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = epoch_valid_g_loss
        torch.save(encoder, os.path.join(opt.save_dir, 'encoder_%03d.pth' % (epoch)))
        torch.save(decoder, os.path.join(opt.save_dir, 'decoder_%03d.pth' % (epoch)))
        torch.save(discriminator, os.path.join(opt.save_dir, 'discriminator_%03d.pth' % (epoch)))
        
    writer.add_scalars('Training Loss 50 Retina', {
            'Generator': epoch_g_loss / len(trainloader),
            'Discriminator': epoch_d_loss / len(trainloader),
            'Topology': epoch_top_loss / len(trainloader)
        }, epoch)

    writer.add_scalars('Valid Loss 50 Retina', {
            'Generator': epoch_valid_g_loss,
            'Topology': epoch_valid_top_loss / len(validloader)
        }, epoch)

    writer.add_text('Best Epoch 50 Retina', str(best_epoch))
