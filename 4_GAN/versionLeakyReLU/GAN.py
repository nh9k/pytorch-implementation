import torch.nn as nn
import numpy as np
import math


class Generator(nn.Module):
    def __init__(self, latent_vector):
        super(Generator, self).__init__()
        self.latent_vector = latent_vector
        self.imageshape = (1, 28, 28)
        self.myG = nn.Sequential(
            nn.Linear(self.latent_vector, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(self.imageshape))),
            nn.Tanh()
        )
        # for m in self.modules():
        #     if type(m) == nn.Linear:
        #         '''
        #         m.weight.data.normal_(0.0, 0.1)     # 가중치를 평균 0, 편차 0.1로 초기화
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화
        #         '''
        #         '''
        #         nn.init.xavier_normal(m.weight.data)   # Xavier Initialization # Sigmoid / Tanh
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화
        #         '''
        #         nn.init.kaiming_normal_(m.weight.data) # Kaming He Initialization # ReLU
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화

    def forward(self, x):
        x = self.myG(x)
        x = x.view(x.size(0), *self.imageshape)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_vector):
        super(Discriminator, self).__init__()
        self.latent_vector = latent_vector
        self.imageshape = (1, 28, 28)
        self.myD = nn.Sequential(
            nn.Linear(int(np.prod(self.imageshape)), 512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(True),
            # nn.Linear(1024,512),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, self.latent_vector),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.latent_vector, 1),
            nn.Sigmoid()
        )
        # for m in self.modules():
        #     if type(m) == nn.Linear:
        #         '''
        #         m.weight.data.normal_(0.0, 0.1)     # 가중치를 평균 0, 편차 0.1로 초기화
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화
        #         '''
        #
        #         nn.init.xavier_normal_(m.weight.data)   # Xavier Initialization # Sigmoid / Tanh
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화
        #
        #         nn.init.kaiming_normal_(m.weight.data) # Kaming He Initialization # ReLU
        #         m.bias.data.fill_(0)                # 편차를 0으로 초기화

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.myD(x)
        return x

