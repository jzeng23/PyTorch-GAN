import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from topologylayer.nn import LevelSetLayer2D, SumBarcodeLengths

# initializing variables
parser = argparse.ArgumentParser()
parser.add_argument("--best_epoch", type=str, default="352", help="epoch of model to test")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--a", type=float, default=50, help="weight of top_loss")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=782, help="interval between image sampling")
parser.add_argument("--save_dir", type=str, default='best_model', help="directory where you save models")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.model = nn.ModuleList([self.model])


        self.mu = nn.ModuleList([nn.Linear(512, opt.latent_dim)])
        self.logvar = nn.ModuleList([nn.Linear(512, opt.latent_dim)])

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

os.makedirs("../../../data/mnist", exist_ok=True)
dataset = datasets.MNIST(
        "../../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    )

test, _ = random_split(dataset, [10000, 50000])
testloader = torch.utils.data.DataLoader(test, batch_size=opt.batch_size, shuffle=False)

adversarial_loss = torch.nn.BCELoss(reduction='mean')
pixelwise_loss = torch.nn.L1Loss()

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

levelsetlayer = LevelSetLayer2D(size=(opt.img_size, opt.img_size), maxdim=1, sublevel=False)
beta0layer = SumBarcodeLengths(dim=0)

encoder = torch.load('../best_model_50_small/encoder_%s.pth' % opt.best_epoch)
decoder = torch.load('../best_model_50_small/decoder_%s.pth' % opt.best_epoch)
discriminator = torch.load('../best_model_50_small/discriminator_%s.pth' % opt.best_epoch)

total_g_loss = 0
total_top_loss = 0
total_d_loss = 0

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

encoder.eval()
decoder.eval()
discriminator.eval()

for i, (imgs, _) in enumerate(testloader):
    # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    
    # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        top_loss = 0
        for n in range(decoded_imgs.size(0)):
            barcode = levelsetlayer(decoded_imgs[n, 0, :, :])
            top_loss += beta0layer(barcode)
        top_loss = top_loss / (opt.a*decoded_imgs.size(0))
        total_top_loss += top_loss.item()

        g_loss = top_loss + 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )
        total_g_loss += g_loss.item()

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).cuda()
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)
        total_d_loss += d_loss.item()

mean_g_loss = total_g_loss / len(testloader)
mean_top_loss = total_top_loss / len(testloader)
mean_d_loss = total_d_loss / len(testloader)
print('a = ', opt.a)
print('Generator Loss: ', mean_g_loss)
print('Topology Loss: ', mean_top_loss)
print('Discriminator Loss: ', mean_d_loss)
