from GAN import Generator, Discriminator
import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=60, help='batch size,  default: 60')
    parser.add_argument('--num_workers', default=4, help='num workers,  default: 4')
    parser.add_argument('--learning_rate', default=0.0002, help='learning rate,  default: 0.0002')
    parser.add_argument('--b1', default=0.5, help='adam: decay of first order momentum of gradient,  default: 0.5')
    parser.add_argument('--b2', default=0.9, help='adam: decay of second order momentum of gradient,  default: 0.9')
    parser.add_argument('--epoch', default=100, help='epoch,  default: 100')
    parser.add_argument('--latent_vector',  default=100, help='latent vector z,  default: 100')
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples, default: 1000")
    args = parser.parse_args()

    # #define: Data Loader, way1
    transform = transforms.Compose(
         [transforms.ToTensor(), #image2tensor
          transforms.Normalize((0.5,), (0.5,))]) #normalize using average, root var, if (0.5),(0.5),(0.5) = 3dimension
    #type((0.5))  # <type 'float'> #type((0.5,))  # <type 'tuple'>

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) #num_workers is data load multi processing

    # #define: Data Loader, way2
    # os.makedirs("../../data/mnist", exist_ok=True)
    # trainloader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "../../data/mnist",
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    #         ),
    #     ),
    #     batch_size=args.batch_size,
    #     shuffle=True,
    # )
    #define: model

    G = Generator(args.latent_vector)
    D = Discriminator(args.latent_vector)

    #use: GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    G.to(device)
    D.to(device)

    #define: Optimizer
    adversarial_loss = nn.BCELoss()
    G_optimizer = optim.Adam(G.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.learning_rate, betas=(args.b1, args.b2))

    #for tensorboard
    writer = SummaryWriter(comment='GAN')

    #training
    for epoch in range(args.epoch):
        for i, data in enumerate(trainloader):
            #real images from MNIST
            real_imgs, _ = data
            real_imgs = real_imgs.to(device)

            #fake images from generator
            #z = Variable(torch.Tensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_vector))))
            z = torch.randn(real_imgs.size(0),args.latent_vector).to(device)
            fake_imgs = G(z)

            #Correct answer, real is 1, fake is 0, same as labels
            target_real = torch.ones(real_imgs.size(0), 1, requires_grad=False).to(device)
            target_fake = torch.zeros(real_imgs.size(0), 1, requires_grad=False).to(device)

            #train Discriminator
            D_optimizer.zero_grad()
            D_real_loss = adversarial_loss(D(real_imgs), target_real)
            D_fake_loss = adversarial_loss(D(fake_imgs.detach()), target_fake) #.detach()
            D_loss = (D_real_loss + D_fake_loss)/2
            D_loss.backward() #retain_graph=True
            D_optimizer.step()

            #train Generator
            G_optimizer.zero_grad()
            G_loss = adversarial_loss(D(fake_imgs), target_real)
            G_loss.backward()
            G_optimizer.step()


            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epoch, i, len(trainloader), D_loss.item(), G_loss.item())
            )

            #recording events by tensorboard
            writer.add_scalar('D Loss/train', D_loss.item(), epoch * len(trainloader) + i)
            writer.add_scalar('G_loss/train', G_loss.item(), epoch * len(trainloader) + i)

            batches_done = epoch * len(trainloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    writer.close()
    print('Finished Training')
    torch.save(G.state_dict(), './' +'GAN_Generator' + '.pth')
    torch.save(G.state_dict(), './' +'GAN_Discriminator' + '.pth')