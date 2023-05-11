# Description: Contains Class for the denoising autoencoder architecture
# author: Kolade Gideon @Allaye
# github: www.github.com/allaye
# created: 2023-03-18
# last modified: 2023-03-25
import torch
import torch.functional as F
from torch import nn


class Denoiser(nn.Module):
    """
    Denoiser class for the demonising autoencoder
    """

    def __init__(self):
        super(Denoiser, self).__init__()

        # encoder
        self.en_conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.en_conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.en_conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.LeakyReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.en_2hid = nn.Linear(32 * 16 * 16, 250)
        self.bn4 = nn.BatchNorm1d(250)
        self.en_hid2mu = nn.Linear(250, 20)
        self.en_hid_2sigma = nn.Linear(250, 20)

        # decoder
        self.de_hid2out = nn.Linear(20, 250)
        self.bn5 = nn.BatchNorm1d(250)
        self.de_out2hid = nn.Linear(250, 32 * 16 * 16)
        self.bn6 = nn.BatchNorm1d(32 * 16 * 16)
        self.de_conv1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.de_conv2 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.de_conv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(3)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        x = self.pool(self.relu(self.bn1(self.en_conv1(x))))
        x = self.pool(self.relu(self.bn2(self.en_conv2(x))))
        x = self.pool(self.relu(self.bn3(self.en_conv3(x))))
        x = x.view(-1, 32 * 16 * 16)
        x = self.relu(self.bn4(self.en_2hid(x)))
        mu = self.en_hid2mu(x)
        sigma = self.en_hid_2sigma(x)
        return mu, sigma

    def decoder(self, x):
        x = self.relu(self.bn5(self.de_hid2out(x)))
        x = self.relu(self.bn6(self.de_out2hid(x)))
        x = x.view(-1, 32, 16, 16)
        # x = x.unsqueeze(0) # add batch and channel dimensions
        x = self.unpool(self.relu(self.bn7(self.de_conv1(x))))
        x = self.unpool(self.relu(self.bn8(self.de_conv2(x))))
        x = self.unpool(self.relu(self.bn9(self.de_conv3(x))))
        return self.sigmoid(x.squeeze(0).squeeze(0))

    def forward(self, x):
        mu, sigma = self.encoder(x)
        epsilon = torch.randn_like(sigma)
        z_new = mu + sigma * epsilon
        x_reconstructed = self.decoder(z_new)
        return x_reconstructed, mu, sigma

    def loss_optimizer(self, lr=0.001):
        '''

        :param lr: karpathy constant
        :return:
        '''

        loss_fn = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return loss_fn, optimizer


# ARCHITECTURE
# TRAINING DATA
#
#
# model = nn.Sequential(
#     nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
#     # nn.ReLU(),
#     # nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
#     # nn.ReLU(),
#     # nn.MaxPool2d(kernel_size=2, stride=2),
#     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
#     # nn.ReLU(),
#     # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True),
#     nn.Flatten(),
#     nn.Linear(32 * 30 * 30, 120)
#     # nn.ReLU(),
#     # nn.Linear(1024, 32 * 32 * 32)
# )
#
# enmodel = nn.Sequential(
#     nn.Linear(120, 32 * 30 * 30),
#     nn.Unflatten(1, (32, 30, 30)),
#     # nn.MaxUnpool2d(kernel_size=2, stride=2)
#     nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
#     # nn.MaxUnpool2d(kernel_size=2, stride=2),
#     nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
#     # nn.MaxUnpool2d(kernel_size=2, stride=2),
#     nn.ConvTranspose2d(128, 3, kernel_size=3, stride=1, padding=1),
#     nn.Sigmoid()
#     # nn.Unflatten()
# )
#
# out = model(torch.rand(2, 3, 30, 30))
# print('checking the shape of the output')
# print(out.shape)
# inn = enmodel(out)
# # print('checking the shape of the input')
# print(inn.shape)
# print('checking the output')
#
# denoiser = Denoiser()
# x_reconstructed, mu, sigma = denoiser(torch.rand(1, 3, 360, 240))
# print('mu shape', mu.shape)
# print('x_res', x_reconstructed.shape)
# print("sigma", sigma.shape)

# RuntimeError: mat1 and mat2 shapes cannot be multiplied (128x4 and 512x120)
